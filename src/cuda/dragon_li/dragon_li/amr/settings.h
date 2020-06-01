#pragma once

#include <dragon_li/util/settings.h>

namespace dragon_li {
namespace amr {

template< 
			typename _Settings,
			typename _Types,
			typename _Types::SizeType _GRID_REFINE_SIZE,
			typename _Types::SizeType _GRID_REFINE_X,
			typename _Types::SizeType _GRID_REFINE_Y
		>
class Settings : public _Settings{

public:
	typedef _Types Types;
	typedef typename Types::DataType DataType;
	typedef typename Types::SizeType SizeType;

	static const SizeType GRID_REFINE_SIZE = _GRID_REFINE_SIZE;
	static const SizeType GRID_REFINE_X = _GRID_REFINE_X;
	static const SizeType GRID_REFINE_Y = _GRID_REFINE_Y;
	static const SizeType GRID_REFINE_Z = GRID_REFINE_SIZE / GRID_REFINE_X / GRID_REFINE_Y;
};

}
}
