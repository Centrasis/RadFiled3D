#pragma once
#include <cmath>
#include <limits>
#include <type_traits>


namespace RadFiled3D {
	/** Integer arithmetic that clamps to the representable range of T instead of wrapping around.
	* Used to join integral layers (e.g. 8-bit masks), where a wrapped value would silently corrupt the data.
	*/
	namespace SaturatingArithmetic {
		template<typename T>
		constexpr T add(T a, T b) {
			static_assert(std::is_integral_v<T>);
			if constexpr (std::is_signed_v<T>) {
				if (b > 0 && a > std::numeric_limits<T>::max() - b)
					return std::numeric_limits<T>::max();
				if (b < 0 && a < std::numeric_limits<T>::min() - b)
					return std::numeric_limits<T>::min();
			}
			else {
				if (a > std::numeric_limits<T>::max() - b)
					return std::numeric_limits<T>::max();
			}
			return static_cast<T>(a + b);
		}

		template<typename T>
		constexpr T subtract(T a, T b) {
			static_assert(std::is_integral_v<T>);
			if constexpr (std::is_signed_v<T>) {
				if (b < 0 && a > std::numeric_limits<T>::max() + b)
					return std::numeric_limits<T>::max();
				if (b > 0 && a < std::numeric_limits<T>::min() + b)
					return std::numeric_limits<T>::min();
			}
			else {
				if (a < b)
					return 0;
			}
			return static_cast<T>(a - b);
		}

		template<typename T>
		constexpr T multiply(T a, T b) {
			static_assert(std::is_integral_v<T>);
			if (a == 0 || b == 0)
				return 0;
			constexpr T max = std::numeric_limits<T>::max();
			constexpr T min = std::numeric_limits<T>::min();
			if constexpr (std::is_signed_v<T>) {
				const bool positive = (a > 0) == (b > 0);
				if (positive) {
					if (a > 0 ? a > max / b : a < max / b)
						return max;
				}
				else {
					if (a > 0 ? b < min / a : a < min / b)
						return min;
				}
			}
			else {
				if (a > max / b)
					return max;
			}
			return static_cast<T>(a * b);
		}

		/** Division by zero saturates towards the sign of the dividend (0 / 0 yields 0). */
		template<typename T>
		constexpr T divide(T a, T b) {
			static_assert(std::is_integral_v<T>);
			if (b == 0) {
				if (a == 0)
					return 0;
				if constexpr (std::is_signed_v<T>)
					return a > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
				else
					return std::numeric_limits<T>::max();
			}
			if constexpr (std::is_signed_v<T>) {
				if (a == std::numeric_limits<T>::min() && b == -1)
					return std::numeric_limits<T>::max();
			}
			return static_cast<T>(a / b);
		}

		/** Mean of a and b without an overflowing intermediate sum. Exact (truncated towards zero) for types narrower
		* than 64 bit; for 64-bit types the result may deviate by one from the truncated mean.
		*/
		template<typename T>
		constexpr T mean(T a, T b) {
			static_assert(std::is_integral_v<T>);
			if constexpr (sizeof(T) < sizeof(long long)) {
				using Wide = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;
				return static_cast<T>((static_cast<Wide>(a) + static_cast<Wide>(b)) / 2);
			}
			const T half_sum = static_cast<T>(a / 2 + b / 2);
			const T remainder_sum = static_cast<T>(a % 2 + b % 2);
			return static_cast<T>(half_sum + remainder_sum / 2);
		}

		/** a * (1 - ratio) + b * ratio, rounded to nearest and clamped to the range of T. */
		template<typename T>
		T blend(T a, T b, float ratio) {
			static_assert(std::is_integral_v<T>);
			const long double value = std::round(static_cast<long double>(a) * (1.0L - ratio) + static_cast<long double>(b) * ratio);
			if (!(value > static_cast<long double>(std::numeric_limits<T>::min())))
				return std::numeric_limits<T>::min();
			if (!(value < static_cast<long double>(std::numeric_limits<T>::max())))
				return std::numeric_limits<T>::max();
			return static_cast<T>(value);
		}
	}
}
