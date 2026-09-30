#pragma once
#include <map>
#include "radfiled3d/storage/types.hpp"


namespace radfiled3d {
	namespace storage {
		class BasicFieldStore;

		class Registry {
		private:
			static std::map<radfiled3d::storage::StoreVersion, std::unique_ptr<radfiled3d::storage::BasicFieldStore>> registered_field_stores;

		public:
			static void register_store(radfiled3d::storage::StoreVersion version, std::unique_ptr<radfiled3d::storage::BasicFieldStore> store);

			static const radfiled3d::storage::BasicFieldStore* get_store_by(radfiled3d::storage::StoreVersion version);
			static const radfiled3d::storage::StoreVersion get_highest_supported_version_by(const std::string& version_str);
			static const radfiled3d::storage::StoreVersion get_lowest_supported_version_by(const std::string& version_str);
			static const radfiled3d::storage::BasicFieldStore* get_highest_supported_store(const std::string& version_str);
			static const radfiled3d::storage::BasicFieldStore* get_lowest_supported_store(const std::string& version_str);

			static radfiled3d::storage::StoreVersion get_highest_registered_version();
		};
	};
};
	