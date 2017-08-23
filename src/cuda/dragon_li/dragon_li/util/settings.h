#pragma once

#include <dragon_li/util/types.h>

namespace dragon_li {
namespace util {

template< 
			typename _Types,
			int _THREADS,
			int _CTAS,
			int _CDP_THREADS_BITS,
			int _CDP_THRESHOLD
		>
class Settings {

public:
	typedef _Types Types;
	typedef typename _Types::SizeType SizeType;

	static const SizeType THREADS = _THREADS;
	static const SizeType CTAS = _CTAS;
	static const SizeType CDP_THREADS_BITS = _CDP_THREADS_BITS;
	static const SizeType CDP_THREADS = 1 << CDP_THREADS_BITS;
	static const SizeType CDP_THRESHOLD = _CDP_THRESHOLD;

};

}
}
