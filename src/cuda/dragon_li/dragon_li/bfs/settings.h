#pragma once

#include <dragon_li/util/settings.h>

namespace dragon_li {
namespace bfs {

template< 
			typename _Settings,
			typename _Types,
			int _MASK_BITS
		>
class Settings : public _Settings{

public:
	typedef _Types Types;
	typedef typename _Types::VertexIdType VertexIdType;
	typedef typename _Types::EdgeWeightType EdgeWeightType;
	typedef typename _Types::MaskType MaskType;
	typedef typename _Settings::SizeType SizeType;

	static const SizeType MASK_BITS = _MASK_BITS;
	static const SizeType MASK_SIZE = 1 << MASK_BITS;
	static const SizeType MASK_MASK = MASK_SIZE - 1;

};

}
}
