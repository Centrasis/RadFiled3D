#pragma once
#include <map>
#include "RadFiled3D/storage/Types.hpp"


namespace RadFiled3D {
	namespace Storage {
		class BasicFieldStore;

		class Registry {
		private:
			static std::map<RadFiled3D::Storage::StoreVersion, std::unique_ptr<RadFiled3D::Storage::BasicFieldStore>> registered_field_stores;

		public:
			static void register_store(RadFiled3D::Storage::StoreVersion version, std::unique_ptr<RadFiled3D::Storage::BasicFieldStore> store);

			static const RadFiled3D::Storage::BasicFieldStore* get_store_by(RadFiled3D::Storage::StoreVersion version);
			static const RadFiled3D::Storage::StoreVersion get_highest_supported_version_by(const std::string& version_str);
			static const RadFiled3D::Storage::StoreVersion get_lowest_supported_version_by(const std::string& version_str);
			static const RadFiled3D::Storage::BasicFieldStore* get_highest_supported_store(const std::string& version_str);
			static const RadFiled3D::Storage::BasicFieldStore* get_lowest_supported_store(const std::string& version_str);

			static RadFiled3D::Storage::StoreVersion get_highest_registered_version();
		};
	};
};
	