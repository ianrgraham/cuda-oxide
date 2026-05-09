/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, PinnedHostBuffer};

#[test]
fn pinned_host_buffer_exposes_initialized_slice() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut host = PinnedHostBuffer::<u32>::new(&ctx, 4).expect("failed to allocate pinned host");

    assert_eq!(host.len(), 4);
    assert!(!host.is_empty());
    assert_eq!(host.as_slice(), &[0, 0, 0, 0]);

    host.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
    assert_eq!(&host[..], &[1, 2, 3, 4]);
}

#[test]
fn pinned_host_buffer_supports_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let host = PinnedHostBuffer::<u32>::new(&ctx, 0).expect("failed to create empty pinned host");

    assert_eq!(host.len(), 0);
    assert!(host.is_empty());
    assert_eq!(host.as_slice(), &[]);
}

#[test]
fn pinned_host_buffer_roundtrips_through_device_buffer() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let input = PinnedHostBuffer::from_slice(&ctx, &[1_u32, 2, 3, 4])
        .expect("failed to allocate pinned input");
    let device =
        DeviceBuffer::from_pinned_host(&stream, &input).expect("failed to copy input to device");
    let mut output =
        PinnedHostBuffer::<u32>::new(&ctx, input.len()).expect("failed to allocate pinned output");

    device
        .copy_to_pinned_host(&stream, &mut output)
        .expect("failed to copy output to host");

    assert_eq!(output.as_slice(), input.as_slice());
}
