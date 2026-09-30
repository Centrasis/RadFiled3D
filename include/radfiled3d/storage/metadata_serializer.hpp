#pragma once
#include "radfiled3d/storage/types.hpp"
#include <memory>

namespace radfiled3d {
	namespace storage {
		class MetadataSerializer {
		public:
			virtual ~MetadataSerializer() = default;
			virtual void serializeMetadata(std::ostream& buffer, std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> metadata) const = 0;
		};

		namespace v1 {
			class MetadataSerializer : public radfiled3d::storage::MetadataSerializer {
			public:
				MetadataSerializer() = default;
				virtual void serializeMetadata(std::ostream& buffer, std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> metadata) const override;
			};
		};
	}
}