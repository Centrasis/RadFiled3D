#include "radfiled3d/storage/registry.hpp"
#include "radfiled3d/storage/radiation_field_store.hpp"


std::map<radfiled3d::storage::StoreVersion, std::unique_ptr<radfiled3d::storage::BasicFieldStore>> radfiled3d::storage::Registry::registered_field_stores;


void radfiled3d::storage::Registry::register_store(radfiled3d::storage::StoreVersion version, std::unique_ptr<radfiled3d::storage::BasicFieldStore> store)
{
	auto store_itr = radfiled3d::storage::Registry::registered_field_stores.find(version);
	if (store_itr != radfiled3d::storage::Registry::registered_field_stores.end())
		throw RadiationFieldStoreException(std::string("Can't register two stores for the same version!"));

	radfiled3d::storage::Registry::registered_field_stores.insert({
		version,
		std::move(store)
	});
}

const radfiled3d::storage::BasicFieldStore* radfiled3d::storage::Registry::get_store_by(radfiled3d::storage::StoreVersion version)
{
	auto store_itr = radfiled3d::storage::Registry::registered_field_stores.find(version);
	if (store_itr != radfiled3d::storage::Registry::registered_field_stores.end()) {
		return (*store_itr).second.get();
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version requested: V") + std::to_string((static_cast<char>(version) - static_cast<char>(StoreVersion::V1)) + 1));
}

const radfiled3d::storage::StoreVersion radfiled3d::storage::Registry::get_highest_supported_version_by(const std::string& version_str)
{
	for (auto it = radfiled3d::storage::Registry::registered_field_stores.rbegin(); it != radfiled3d::storage::Registry::registered_field_stores.rend(); ++it) {
		if (it->second->check_version_string_validity(version_str)) {
			return it->first;
		}
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

const radfiled3d::storage::StoreVersion radfiled3d::storage::Registry::get_lowest_supported_version_by(const std::string& version_str)
{
	for (const auto& [key, value] : radfiled3d::storage::Registry::registered_field_stores) {
		if (value->check_version_string_validity(version_str))
			return key;
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

const radfiled3d::storage::BasicFieldStore* radfiled3d::storage::Registry::get_highest_supported_store(const std::string& version_str)
{
	for (auto it = radfiled3d::storage::Registry::registered_field_stores.rbegin(); it != radfiled3d::storage::Registry::registered_field_stores.rend(); ++it) {
		if (it->second->check_version_string_validity(version_str)) {
			return it->second.get();
		}
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

const radfiled3d::storage::BasicFieldStore* radfiled3d::storage::Registry::get_lowest_supported_store(const std::string& version_str)
{
	for (const auto& [key, value] : radfiled3d::storage::Registry::registered_field_stores) {
		if (value->check_version_string_validity(version_str))
			return value.get();
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

radfiled3d::storage::StoreVersion radfiled3d::storage::Registry::get_highest_registered_version() {
	return radfiled3d::storage::Registry::registered_field_stores.rbegin()->first;
}
