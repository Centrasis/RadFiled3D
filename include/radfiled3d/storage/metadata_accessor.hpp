#pragma once
#include "radfiled3d/storage/types.hpp"
#include <memory>

namespace radfiled3d {
	namespace storage {
		class MetadataAccessor {
		public:
			virtual ~MetadataAccessor() = default;
			virtual std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> accessMetadata(std::istream& buffer, bool quick_peek_only = false) const = 0;
			virtual size_t get_metadata_size(std::istream& stream) const = 0;
		};

		namespace v1 {
			class MetadataAccessor : public radfiled3d::storage::MetadataAccessor {
			protected:
				radfiled3d::storage::v1::RadiationFieldMetadata meta_template;
			public:
				virtual std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> accessMetadata(std::istream& buffer, bool quick_peek_only = false) const override;
				virtual size_t get_metadata_size(std::istream& stream) const override;
			};
		};
	}
}