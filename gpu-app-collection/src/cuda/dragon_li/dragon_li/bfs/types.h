#pragma once

namespace dragon_li {
namespace bfs {

template<
	typename _Types,
	typename _VertexIdType,
	typename _EdgeWeightType,
	typename _MaskType
	>
class Types : public _Types {

public:
	typedef _VertexIdType VertexIdType;
	typedef _EdgeWeightType EdgeWeightType;
	typedef _MaskType MaskType;
};

}
}
