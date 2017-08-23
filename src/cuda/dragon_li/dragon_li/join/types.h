#pragma once

namespace dragon_li {
namespace join {

template<
	typename _Types,
	typename _DataType
	>
class Types : public _Types {

public:
	typedef _DataType DataType;
};

}
}
