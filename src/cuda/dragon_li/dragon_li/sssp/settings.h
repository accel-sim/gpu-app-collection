#pragma once

#include <dragon_li/util/settings.h>

namespace dragon_li {
namespace sssp {

template< 
			typename _Settings,
			typename _Types,
			int _INF_WEIGHT
		>
class Settings : public _Settings{

public:
	typedef _Types Types;
	typedef typename _Types::VertexIdType VertexIdType;
	typedef typename _Types::EdgeWeightType EdgeWeightType;
	typedef typename _Settings::SizeType SizeType;

	static const EdgeWeightType INF_WEIGHT = _INF_WEIGHT;
};

}
}
