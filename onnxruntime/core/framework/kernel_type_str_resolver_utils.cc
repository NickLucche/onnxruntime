// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/framework/kernel_type_str_resolver_utils.h"

#include "core/common/common.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/optimizer/layout_transformation/layout_transformation_potentially_added_ops.h"
#include "flatbuffers/flatbuffers.h"

namespace onnxruntime::kernel_type_str_resolver_utils {

static constexpr auto* kStandaloneKernelTypeStrResolverFileIdentifier = "ktsr";

#if !defined(ORT_MINIMAL_BUILD)

gsl::span<const OpIdentifierWithStringViews> GetLayoutTransformationRequiredOpIdentifiers() {
  return kLayoutTransformationPotentiallyAddedOps;
}

Status SaveKernelTypeStrResolverToBuffer(const KernelTypeStrResolver& kernel_type_str_resolver,
                                         flatbuffers::DetachedBuffer& buffer, gsl::span<const uint8_t>& buffer_span) {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<fbs::KernelTypeStrResolver> fbs_kernel_type_str_resolver;
  ORT_RETURN_IF_ERROR(kernel_type_str_resolver.SaveToOrtFormat(builder, fbs_kernel_type_str_resolver));
  builder.Finish(fbs_kernel_type_str_resolver, kStandaloneKernelTypeStrResolverFileIdentifier);
  buffer = builder.Release();
  buffer_span = gsl::make_span(buffer.data(), buffer.size());
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

Status LoadKernelTypeStrResolverFromBuffer(KernelTypeStrResolver& kernel_type_str_resolver,
                                           gsl::span<const uint8_t> buffer_span) {
  flatbuffers::Verifier verifier{buffer_span.data(), buffer_span.size_bytes()};
  ORT_RETURN_IF_NOT(verifier.VerifyBuffer<fbs::KernelTypeStrResolver>(kStandaloneKernelTypeStrResolverFileIdentifier),
                    "Failed to verify KernelTypeStrResolver flatbuffers data.");
  const auto* fbs_kernel_type_str_resolver = flatbuffers::GetRoot<fbs::KernelTypeStrResolver>(buffer_span.data());
  ORT_RETURN_IF_ERROR(kernel_type_str_resolver.LoadFromOrtFormat(*fbs_kernel_type_str_resolver));
  return Status::OK();
}

Status AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(KernelTypeStrResolver& kernel_type_str_resolver) {
  KernelTypeStrResolver resolver_with_required_ops{};

  // to generate kLayoutTransformationRequiredOpsKernelTypeStrResolverBytes, run the test:
  //   KernelTypeStrResolverUtilsTest.DISABLED_PrintExpectedLayoutTransformationRequiredOpsResolverByteArray

  // clang-format off
  constexpr uint8_t kLayoutTransformationRequiredOpsKernelTypeStrResolverBytes[] = {
      0x10, 0x00, 0x00, 0x00, 0x6b, 0x74, 0x73, 0x72, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00,
      0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0xbc, 0x06, 0x00, 0x00,
      0x4c, 0x02, 0x00, 0x00, 0xe0, 0x01, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x00, 0x14, 0x06, 0x00, 0x00,
      0x88, 0x01, 0x00, 0x00, 0xb8, 0x05, 0x00, 0x00, 0x1c, 0x05, 0x00, 0x00, 0x18, 0x07, 0x00, 0x00,
      0xcc, 0x04, 0x00, 0x00, 0x0c, 0x01, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00, 0x54, 0x05, 0x00, 0x00,
      0x3c, 0x06, 0x00, 0x00, 0xf8, 0x02, 0x00, 0x00, 0x7c, 0x02, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x38, 0x03, 0x00, 0x00, 0xec, 0xf8, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x63, 0x6f, 0x6d, 0x2e,
      0x6d, 0x69, 0x63, 0x72, 0x6f, 0x73, 0x6f, 0x66, 0x74, 0x3a, 0x4e, 0x68, 0x77, 0x63, 0x4d, 0x61,
      0x78, 0x50, 0x6f, 0x6f, 0x6c, 0x3a, 0x31, 0x00, 0x20, 0xf9, 0xff, 0xff, 0xf0, 0x06, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x0e, 0xf9, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x08, 0xf9, 0xff, 0xff, 0x44, 0xf9, 0xff, 0xff,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00, 0x3a, 0x54, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x6f, 0x73, 0x65, 0x3a, 0x31,
      0x00, 0x00, 0x00, 0x00, 0x6c, 0xf9, 0xff, 0xff, 0xa4, 0x06, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x5a, 0xf9, 0xff, 0xff,
      0x00, 0x00, 0x00, 0x01, 0x54, 0xf9, 0xff, 0xff, 0x90, 0xf9, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
      0x3a, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x3a, 0x31, 0x00, 0xb4, 0xf9, 0xff, 0xff,
      0x5c, 0x06, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0xa2, 0xf9, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x9c, 0xf9, 0xff, 0xff,
      0xd8, 0xf9, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x34, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x3a, 0x53, 0x71, 0x75,
      0x65, 0x65, 0x7a, 0x65, 0x3a, 0x31, 0x33, 0x00, 0x00, 0xfa, 0xff, 0xff, 0xb4, 0x01, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x48, 0xfa, 0xff, 0xff,
      0x01, 0x00, 0x00, 0x00, 0x1c, 0xfa, 0xff, 0xff, 0xf4, 0x05, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0a, 0xfa, 0xff, 0xff,
      0x00, 0x00, 0x00, 0x01, 0x04, 0xfa, 0xff, 0xff, 0x40, 0xfa, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
      0x3a, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x3a, 0x31, 0x34, 0x00, 0x00, 0x00, 0x00,
      0x68, 0xfa, 0xff, 0xff, 0x3c, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x56, 0xfa, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
      0x50, 0xfa, 0xff, 0xff, 0x8c, 0xfa, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
      0x3a, 0x47, 0x61, 0x74, 0x68, 0x65, 0x72, 0x3a, 0x31, 0x33, 0x00, 0x00, 0xb4, 0xfa, 0xff, 0xff,
      0x00, 0x05, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xfc, 0xfa, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0xd0, 0xfa, 0xff, 0xff, 0x40, 0x05, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xbe, 0xfa, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0xb8, 0xfa, 0xff, 0xff, 0xf4, 0xfa, 0xff, 0xff,
      0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
      0x14, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3a, 0x47, 0x61, 0x74, 0x68, 0x65, 0x72, 0x3a,
      0x31, 0x31, 0x00, 0x00, 0x1c, 0xfb, 0xff, 0xff, 0x98, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x64, 0xfb, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
      0x38, 0xfb, 0xff, 0xff, 0xd8, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x26, 0xfb, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
      0x20, 0xfb, 0xff, 0xff, 0x5c, 0xfb, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
      0x3a, 0x55, 0x6e, 0x73, 0x71, 0x75, 0x65, 0x65, 0x7a, 0x65, 0x3a, 0x31, 0x33, 0x00, 0x00, 0x00,
      0x88, 0xfb, 0xff, 0xff, 0x88, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x76, 0xfb, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
      0x70, 0xfb, 0xff, 0xff, 0xac, 0xfb, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x61, 0x78, 0x65, 0x73, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x00, 0xfc, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0xd4, 0xfb, 0xff, 0xff,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
      0x0d, 0x00, 0x00, 0x00, 0x3a, 0x55, 0x6e, 0x73, 0x71, 0x75, 0x65, 0x65, 0x7a, 0x65, 0x3a, 0x31,
      0x31, 0x00, 0x00, 0x00, 0xfc, 0xfb, 0xff, 0xff, 0x14, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xea, 0xfb, 0xff, 0xff,
      0x00, 0x00, 0x00, 0x01, 0xe4, 0xfb, 0xff, 0xff, 0x20, 0xfc, 0xff, 0xff, 0x28, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x38, 0x01, 0x00, 0x00, 0xdc, 0x00, 0x00, 0x00,
      0xa8, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00,
      0x48, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x63, 0x6f, 0x6d, 0x2e, 0x6d, 0x69, 0x63, 0x72,
      0x6f, 0x73, 0x6f, 0x66, 0x74, 0x3a, 0x51, 0x4c, 0x69, 0x6e, 0x65, 0x61, 0x72, 0x43, 0x6f, 0x6e,
      0x76, 0x3a, 0x31, 0x00, 0x6c, 0xfc, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x54, 0x34, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xbc, 0xfc, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x90, 0xfc, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x79, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x65, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xe4, 0xfc, 0xff, 0xff, 0x06, 0x00, 0x00, 0x00,
      0xb8, 0xfc, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
      0x78, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x65, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x0c, 0xfd, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0xe0, 0xfc, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x54, 0x33, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xd6, 0xfc, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
      0x3c, 0xfd, 0xff, 0xff, 0x07, 0x00, 0x00, 0x00, 0x10, 0xfd, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x54, 0x32, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x64, 0xfd, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
      0x6c, 0xfd, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0x40, 0xfd, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x77, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x65, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x94, 0xfd, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
      0x68, 0xfd, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x54, 0x31, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xbc, 0xfd, 0xff, 0xff, 0x02, 0x00, 0x00, 0x00, 0x58, 0xfd, 0xff, 0xff, 0x94, 0xfd, 0xff, 0xff,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
      0x0b, 0x00, 0x00, 0x00, 0x3a, 0x53, 0x71, 0x75, 0x65, 0x65, 0x7a, 0x65, 0x3a, 0x31, 0x31, 0x00,
      0xb8, 0xfd, 0xff, 0xff, 0x58, 0x02, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xa6, 0xfd, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
      0xa0, 0xfd, 0xff, 0xff, 0xdc, 0xfd, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x3a, 0x49, 0x64, 0x65,
      0x6e, 0x74, 0x69, 0x74, 0x79, 0x3a, 0x31, 0x39, 0x00, 0x00, 0x00, 0x00, 0x04, 0xfe, 0xff, 0xff,
      0xa0, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0xf2, 0xfd, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0xec, 0xfd, 0xff, 0xff,
      0x28, 0xfe, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x18, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x3a, 0x54, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x6f,
      0x73, 0x65, 0x3a, 0x31, 0x33, 0x00, 0x00, 0x00, 0x50, 0xfe, 0xff, 0xff, 0xc0, 0x01, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x3e, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x38, 0xfe, 0xff, 0xff, 0x74, 0xfe, 0xff, 0xff,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00, 0x3a, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x3a, 0x31, 0x36,
      0x00, 0x00, 0x00, 0x00, 0x9c, 0xfe, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x56, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x92, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x8c, 0xfe, 0xff, 0xff,
      0xc8, 0xfe, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x3a, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74,
      0x79, 0x3a, 0x31, 0x33, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xfe, 0xff, 0xff, 0x20, 0x01, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xde, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0xd8, 0xfe, 0xff, 0xff, 0x14, 0xff, 0xff, 0xff,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00, 0x3a, 0x55, 0x6e, 0x73, 0x71, 0x75, 0x65, 0x65, 0x7a, 0x65, 0x3a, 0x31,
      0x00, 0x00, 0x00, 0x00, 0x3c, 0xff, 0xff, 0xff, 0xd4, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x2a, 0xff, 0xff, 0xff,
      0x00, 0x00, 0x00, 0x01, 0x24, 0xff, 0xff, 0xff, 0x60, 0xff, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
      0x09, 0x00, 0x00, 0x00, 0x3a, 0x47, 0x61, 0x74, 0x68, 0x65, 0x72, 0x3a, 0x31, 0x00, 0x00, 0x00,
      0x88, 0xff, 0xff, 0xff, 0x88, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x76, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
      0x70, 0xff, 0xff, 0xff, 0xac, 0xff, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x54, 0x69, 0x6e, 0x64, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0xdc, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3a, 0x53, 0x71, 0x75,
      0x65, 0x65, 0x7a, 0x65, 0x3a, 0x31, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x04, 0x00, 0x08, 0x00,
      0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x54, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
  };
  // clang-format on

  ORT_RETURN_IF_ERROR(LoadKernelTypeStrResolverFromBuffer(resolver_with_required_ops,
                                                          kLayoutTransformationRequiredOpsKernelTypeStrResolverBytes));
  kernel_type_str_resolver.Merge(std::move(resolver_with_required_ops));
  return Status::OK();
}

}  // namespace onnxruntime::kernel_type_str_resolver_utils

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
