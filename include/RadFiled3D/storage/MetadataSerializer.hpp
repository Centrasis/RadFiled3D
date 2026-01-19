#pragma once
#include "RadFiled3D/storage/Types.hpp"
#include <memory>

namespace RadFiled3D {
	namespace Storage {
		class RADFILED_API MetadataSerializer {
		public:
			virtual void serializeMetadata(std::ostream& buffer, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const = 0;
		};

		namespace V1 {
			class RADFILED_API MetadataSerializer : public RadFiled3D::Storage::MetadataSerializer {
			public:
				MetadataSerializer() = default;
				virtual void serializeMetadata(std::ostream& buffer, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const override;
			};
		};
	}
}