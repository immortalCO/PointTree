function run() {
	for ((i=0;i<1;++i)); do
		prog=$1
		g++ $prog.cpp -o ./$prog -O2 -std=c++14 -Wall || (echo "Compilation Error!" >&2 && break);
		shift
		./$prog $@ || (echo "Runtime Error $?" >&2);
	done
}
