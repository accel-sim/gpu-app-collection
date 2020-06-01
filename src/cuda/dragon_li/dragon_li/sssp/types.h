#pragma once

namespace dragon_li {
namespace sssp {

template<
	typename _Types,
	typename _VertexIdType,
	typename _EdgeWeightType
	>
class Types : public _Types {

public:
	typedef _VertexIdType VertexIdType;
	typedef _EdgeWeightType EdgeWeightType;
};

}
}
