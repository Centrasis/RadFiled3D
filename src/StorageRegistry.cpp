#include "RadFiled3D/storage/Registry.hpp"
#include "RadFiled3D/storage/RadiationFieldStore.hpp"


std::map<RadFiled3D::Storage::StoreVersion, std::unique_ptr<RadFiled3D::Storage::BasicFieldStore>> RadFiled3D::Storage::Registry::registered_field_stores;


void RadFiled3D::Storage::Registry::register_store(RadFiled3D::Storage::StoreVersion version, std::unique_ptr<RadFiled3D::Storage::BasicFieldStore> store)
{
	auto store_itr = RadFiled3D::Storage::Registry::registered_field_stores.find(version);
	if (store_itr != RadFiled3D::Storage::Registry::registered_field_stores.end())
		throw RadiationFieldStoreException(std::string("Can't register two stores for the same version!"));

	RadFiled3D::Storage::Registry::registered_field_stores.insert({
		version,
		std::move(store)
	});
}

const RadFiled3D::Storage::BasicFieldStore* RadFiled3D::Storage::Registry::get_store_by(RadFiled3D::Storage::StoreVersion version)
{
	auto store_itr = RadFiled3D::Storage::Registry::registered_field_stores.find(version);
	if (store_itr != RadFiled3D::Storage::Registry::registered_field_stores.end()) {
		return (*store_itr).second.get();
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version requested: V") + std::to_string((static_cast<char>(version) - static_cast<char>(StoreVersion::V1)) + 1));
}

const RadFiled3D::Storage::StoreVersion RadFiled3D::Storage::Registry::get_highest_supported_version_by(const std::string& version_str)
{
	for (auto it = RadFiled3D::Storage::Registry::registered_field_stores.rbegin(); it != RadFiled3D::Storage::Registry::registered_field_stores.rend(); ++it) {
		if (it->second->check_version_string_validity(version_str)) {
			return it->first;
		}
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

const RadFiled3D::Storage::StoreVersion RadFiled3D::Storage::Registry::get_lowest_supported_version_by(const std::string& version_str)
{
	for (const auto& [key, value] : RadFiled3D::Storage::Registry::registered_field_stores) {
		if (value->check_version_string_validity(version_str))
			return key;
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

const RadFiled3D::Storage::BasicFieldStore* RadFiled3D::Storage::Registry::get_highest_supported_store(const std::string& version_str)
{
	for (auto it = RadFiled3D::Storage::Registry::registered_field_stores.rbegin(); it != RadFiled3D::Storage::Registry::registered_field_stores.rend(); ++it) {
		if (it->second->check_version_string_validity(version_str)) {
			return it->second.get();
		}
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

const RadFiled3D::Storage::BasicFieldStore* RadFiled3D::Storage::Registry::get_lowest_supported_store(const std::string& version_str)
{
	for (const auto& [key, value] : RadFiled3D::Storage::Registry::registered_field_stores) {
		if (value->check_version_string_validity(version_str))
			return value.get();
	}

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + version_str);
}

RadFiled3D::Storage::StoreVersion RadFiled3D::Storage::Registry::get_highest_registered_version() {
	return RadFiled3D::Storage::Registry::registered_field_stores.rbegin()->first;
}
