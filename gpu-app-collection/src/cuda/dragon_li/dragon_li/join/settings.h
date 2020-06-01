#pragma once

#include <dragon_li/util/settings.h>

namespace dragon_li {
namespace join {

template< 
			typename _Settings,
			typename _Types,
			int _JOIN_SF,
			int _JOIN_BLOCK_SF
		>
class Settings : public _Settings{

public:
	typedef _Types Types;
	typedef typename Types::DataType DataType;
	typedef typename Types::SizeType SizeType;

	const static SizeType JOIN_SF = _JOIN_SF;
	const static SizeType JOIN_BLOCK_SF = _JOIN_BLOCK_SF;
};

}
}
