// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"

#include "BucketizedBufferAllocator.h"
#include "DmlSubAllocator.h"
// #define PRINT_OUTSTANDING_ALLOCATIONS

namespace Dml
{
    AllocationInfo::~AllocationInfo()
    {
        if (m_owner)
        {
            m_owner->FreeResource(this, m_pooledResourceId);
        }
    }

    BucketizedBufferAllocator::~BucketizedBufferAllocator()
    {
#ifdef PRINT_OUTSTANDING_ALLOCATIONS
        if (!m_outstandingAllocationsById.empty())
        {
            printf("BucketizedBufferAllocator outstanding allocation indices:\n");
            for (auto& entry : m_outstandingAllocationsById)
            {
                printf("%u\n", static_cast<int>(entry.first));
            }
            printf("\n");
        }
#endif
    }

    BucketizedBufferAllocator::BucketizedBufferAllocator(
        ID3D12Device* device,
        std::shared_ptr<ExecutionContext> context,
        std::unique_ptr<DmlSubAllocator>&& subAllocator
        )
        : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)
            )
        ),
        m_device(device),
        m_context(context),
        m_subAllocator(std::move(subAllocator))
    {
    }

    /*static*/ gsl::index BucketizedBufferAllocator::GetBucketIndexFromSize(uint64_t size)
    {
        assert(size != 0);

        // Each bucket is twice as large as the previous one, in ascending order
        gsl::index index = static_cast<gsl::index>(ceil(log2(size)));
        assert((1ull << index) >= size); // This must be true unless there were some strange rounding issues

        // The smallest bucket is 2^n bytes large, where n = c_minResourceSizeExponent
        index = std::max<gsl::index>(index, c_minResourceSizeExponent);
        index -= c_minResourceSizeExponent;

        return index;
    }

    /*static*/ uint64_t BucketizedBufferAllocator::GetBucketSizeFromIndex(gsl::index index)
    {
        return (1ull << (index + c_minResourceSizeExponent));
    }

    void* BucketizedBufferAllocator::Alloc(size_t size)
    {
        return Alloc(size, m_defaultRoundingMode);
    }

    void* BucketizedBufferAllocator::Alloc(size_t size, AllocatorRoundingMode roundingMode)
    {
        // For some reason lotus likes requesting 0 bytes of memory
        size = std::max<size_t>(1, size);

        ComPtr<DmlResourceWrapper> resourceWrapper;
        uint64_t resourceId = 0;

        // Find the bucket for this allocation size
        gsl::index bucketIndex = GetBucketIndexFromSize(size);

        // Some sub allocators have their own rounding mechanisms or alignment requirements of resources
        uint64_t bucketSize = m_subAllocator->ComputeRequiredSize(GetBucketSizeFromIndex(bucketIndex));

        // Use a pooled resource if the size (post rounding, if requested) matches a bucket size
        if (m_defaultRoundingMode == AllocatorRoundingMode::Enabled || size == bucketSize)
        {
            Bucket* bucket = nullptr;

            if (gsl::narrow_cast<gsl::index>(m_pool.size()) <= bucketIndex)
            {
                // Ensure there are sufficient buckets
                m_pool.resize(bucketIndex + 1);
            }

            bucket = &m_pool[bucketIndex];

            if (bucket->resources.empty())
            {
                // No more resources in this bucket - allocate a new one
                resourceWrapper = m_subAllocator->Alloc(bucketSize);
                resourceId = ++m_currentResourceId;
            }
            else
            {
                // Retrieve a resource from the bucket
                resourceWrapper = std::move(bucket->resources.back().resource);
                resourceId = bucket->resources.back().resourceId;
                bucket->resources.pop_back();
            }
        }
        else
        {
            // The allocation will not be pooled.  Construct a new one
            bucketSize = m_subAllocator->ComputeRequiredSize(size);
            resourceWrapper = m_subAllocator->Alloc(bucketSize);
            resourceId = ++m_currentResourceId;
        }

        assert(resourceWrapper != nullptr);
        assert(resourceWrapper->GetResourceInUavState()->GetDesc().Width == bucketSize);

        ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(
            this,
            ++m_currentAllocationId,
            resourceId,
            resourceWrapper.Get(),
            size
        );

    #if _DEBUG
        m_outstandingAllocationsById[allocInfo->GetId()] = allocInfo.Get();
    #endif

        return allocInfo.Detach();
    }

    void BucketizedBufferAllocator::Free(void* p)
    {
        // Release Lotus's reference on the allocation.  The allocation
        // also inherits IUnknown, and once its final reference reaches zero
        // it will call FreeResource
        ComPtr<AllocationInfo> allocInfo;
        allocInfo.Attach(static_cast<AllocationInfo*>(p));
    }

    void BucketizedBufferAllocator::FreeResource(void* p, uint64_t pooledResourceId)
    {
        AllocationInfo *allocInfo = static_cast<AllocationInfo*>(p);

        assert(allocInfo != nullptr); // Can't free nullptr

        if (allocInfo->GetOwner() != this)
        {
            // This allocation doesn't belong to this allocator!
            ORT_THROW_HR(E_INVALIDARG);
        }

        // Free the resource to the pool if its size matches a bucket size
        gsl::index bucketIndex = GetBucketIndexFromSize(allocInfo->GetRequestedSize());
        if (GetBucketSizeFromIndex(bucketIndex) == allocInfo->GetResourceInUavState()->GetDesc().Width)
        {
            if (gsl::narrow_cast<gsl::index>(m_pool.size()) <= bucketIndex)
            {
                // Ensure there are sufficient buckets
                m_pool.resize(bucketIndex + 1);
            }

            // Return the resource to the bucket
            Bucket* bucket = &m_pool[bucketIndex];

            Resource resource = {allocInfo->DetachResourceWrapper(), pooledResourceId};
            bucket->resources.push_back(resource);
        }
        else
        {
            // Free the underlying allocation once queued work has completed.
#ifdef _GAMING_XBOX
            m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->DetachResourceWrapper().Get()).Get());
#else
            m_context->QueueReference(allocInfo->DetachResourceWrapper().Get());
#endif
        }

    #if _DEBUG
        assert(m_outstandingAllocationsById[allocInfo->GetId()] == allocInfo);
        m_outstandingAllocationsById.erase(allocInfo->GetId());
    #endif

        // The allocation info is already destructing at this point
    }


    const AllocationInfo* BucketizedBufferAllocator::DecodeDataHandle(const void* opaqueHandle)
    {
        if (opaqueHandle == nullptr)
        {
            // There is no memory allocated which needs to be decoded.
            ORT_THROW_HR(E_INVALIDARG);
        }
        const auto* allocInfo = static_cast<const AllocationInfo*>(opaqueHandle);

        auto owner = allocInfo->GetOwner();
        //The owner can be null if the resource was wrapped via CreateGPUAllocationFromD3DResource
        if (owner != nullptr && owner != this)
        {
            // This allocation doesn't belong to this allocator!
            ORT_THROW_HR(E_INVALIDARG);
        }

        return allocInfo;
    }

    void BucketizedBufferAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_defaultRoundingMode = roundingMode;
    }

    CPUAllocator::CPUAllocator(OrtMemType memType)
        : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML CPU",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0),
                0,
                memType
            )
        )
    {
    }

    void* CPUAllocator::Alloc(size_t size)
    {
        if (size <= 0)
        {
            return nullptr;
        }
        void* p = malloc(size);
        return p;
    }

    void CPUAllocator::Free(void* p)
    {
        free(p);
    }

} // namespace Dml
