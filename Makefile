default: quantify

quantify_wrap.cxx: quantify_wrap.i quantify.hpp
	DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/Cellar/pcre/8.37/lib/ swig -o quantify_wrap.cxx -c++ -python quantify_wrap.i

quantify: quantify_wrap.cxx quantify.cpp quantify.hpp
	g++ -O3 -fPIC -shared -o _quantify.so -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/ quantify_wrap.cxx quantify.cpp -lpython2.7
