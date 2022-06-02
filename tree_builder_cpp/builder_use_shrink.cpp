#ifndef NOBIND
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
namespace py = pybind11;
#endif

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <ctime>
#include <random>
using namespace std;

#define IL inline
#define len(a) int((a).size())
#define append push_back
#define lbin(x) (31 - __builtin_clz((x)))

#ifndef DEBUG
#include "linalg.h"
#include "dataanalysis.h"
#else
#warning "debug mode on, pca skipped"
#endif

namespace builder_cpp {

	struct P {
		float d[3];
		int i;

		IL float operator - (const P& p) const {
			float ans = 0.0;
			for (int i = 0; i < 3; ++i) ans += (d[i] - p.d[i]) * (d[i] - p.d[i]);
			return ans; 
		}
		IL P operator - () const {
			return (P) {-d[0], -d[1], -d[2], i};
		}
	};
	const P AXIS[3] = {
		(P) {float(1.), float(0.), float(0.), -1},
		(P) {float(0.), float(1.), float(0.), -1},
		(P) {float(0.), float(0.), float(1.), -1}
	};

	IL long init_rand_var() {
		char *x = new char, *z = NULL;
		long r = x - z;
		delete x;
		srand(r);
		return r;
	}

	std::mt19937 random_engine(init_rand_var());

	IL int randint(int l, int r) {
		return uniform_int_distribution<int>(l, r)(random_engine);
	}
	template<class Vector> IL void shuffle(Vector& a) {
		return shuffle(a.begin(), a.end(), random_engine);
	}

	struct Triple {
		int a, b, c;
		IL bool operator < (const Triple& t) const {
			if(a != t.a) return a < t.a;
			if(b != t.b) return b < t.b;
			return c < t.c;
		}
	};

	#define GRID_DEF 0.1
	#define MAX_DIFF_LIM_DEF 2
	#define SYM_THRES 0.5

	class Builder {
	public:
		vector<int> arrange;
		vector<P> p4build;
		bool init = false;
		float RANGE[3];
		P current_part_v;

		IL void part(vector<P> &p, P v) {
			current_part_v = v;
			auto d = v.d;
			auto val = [d] (const P& p) -> float {return d[0] * p.d[0] + d[1] * p.d[1] + d[2] * p.d[2];};
			auto cmp = [val] (const P& a, const P& b) -> bool {return val(a) < val(b);};
			nth_element(p.begin(), p.begin() + len(p) / 2, p.end(), cmp);
		}

		IL void pca_part(vector<P> &p) {
			#ifndef DEBUG
			using namespace alglib;
			real_2d_array pts, v;
			real_1d_array s2;
			pts.setlength(len(p), 3);
			for(int i = 0; i < len(p); ++i)
				for(int j = 0; j < 3; ++j)
					pts[i][j] = p[i].d[j];

			s2.setlength(1);
			v.setlength(3, 1);
			pcatruncatedsubspace(pts, len(p), 3, 1, 0.0, 2, s2, v);
			P div = (P) {(float) v[0][0], (float) v[1][0], (float) v[2][0], -1};

#ifdef NOBIND
// fprintf(stderr, "d = %.4lf %.4lf %.4lf\n", d[0], d[1], d[2]);
#endif

			part(p, v);
			#endif
		}


		bool shrink_symmetry(vector<P>& pts_ref, int d, bool shrink=true, float grid_coef=GRID_DEF, 
			int max_diff_lim=MAX_DIFF_LIM_DEF, int min_n_lim=16, float thres=SYM_THRES) {
			float grid = RANGE[d] * grid_coef;
			auto pts = pts_ref;
			int n = len(pts);
			if (n <= min_n_lim) return false;
			auto cmp = [d] (const P& a, const P& b) -> bool {return a.d[d] < b.d[d];};

			float max_diff = std::max_element(pts.begin(), pts.end(), cmp)->d[d] 
						- std::min_element(pts.begin(), pts.end(), cmp)->d[d];
			if (max_diff < max_diff_lim * grid) return false;

			
			std::nth_element(pts.begin(), pts.begin() + n / 2, pts.end(), cmp);
			auto mid = pts[n / 2].d[d];

			for(auto& p : pts) {
				p.d[d] -= mid;
				if(p.d[d] < 0) p.d[d] = -p.d[d];
			}
			auto l = vector<P>(pts.begin(), pts.begin() + n / 2);
			auto r = vector<P>(pts.begin() + n / 2, pts.end());

			map<Triple, int> count;
			int good = 0;
			for(auto &p : l) ++count[(Triple) {int(p.d[0] / grid), int(p.d[1] / grid), int(p.d[2] / grid)}];
			for(auto &p : r) {
				int &c = count[(Triple) {int(p.d[0] / grid), int(p.d[1] / grid), int(p.d[2] / grid)}];
				if(c) {++good; --c;}
			}
			good *= 2;
#ifdef NOBIND
if (good >= n * thres) {
	fprintf(stderr, "shrink_symmetry n = %d d = %d good = %.4lf max_diff = %d\n", n, d, float(good) / n, int(max_diff / grid));
	fflush(stderr);
}
#endif
			if(good < n * thres) return false;
			if(shrink) pts_ref = pts;
			return true;
		}

		void build(vector<P> p, vector<P> sym_part_v=vector<P>()) {
			int n = len(p);
			if (!init) {
				init = true;
				p4build = p;
				for(int d = 0; d < 3; ++d) {
					auto cmp = [d] (const P& a, const P& b) -> bool {return a.d[d] < b.d[d];};
					auto v = std::max_element(p.begin(), p.end(), cmp)->d[d] 
							- std::min_element(p.begin(), p.end(), cmp)->d[d];
					RANGE[d] = v;
#ifdef NOBIND
fprintf(stderr, "RANGE[%d] = %.6lf\n", d, v);
#endif
				}
			}

			#ifdef NOBIND
			// if (n == 1024) {
			// 	static int counter = 0;
			// 	if(++counter == 2) {
			// 		cout << n << endl;
			// 		for(auto p : p) cout << p.d[0] << ' ' << p.d[1] << ' ' << p.d[2] << endl;
			// 		exit(0);
			// 	}
			// }
			#endif

			if (n == 1) {
				arrange.append(p[0].i);
				return;
			}

			// for(int have = 1, count = 0; have && count <= 5; ) {
			// 	have = 0;
			// 	vector<int> order{0, 1, 2};
			// 	vector<float> max_diff;

			// 	int too_small = 0;

			// 	for(int d = 0; d < 3; ++d) {
			// 		auto cmp = [d] (const P& a, const P& b) -> bool {return a.d[d] < b.d[d];};
			// 		auto v = std::max_element(p.begin(), p.end(), cmp)->d[d] 
			// 				- std::min_element(p.begin(), p.end(), cmp)->d[d];
			// 		max_diff.append(v / RANGE[d]);
			// 		if (v < GRID_DEF * MAX_DIFF_LIM_DEF) ++too_small;
			// 	}
			// 	if(too_small >= 1) break;

			// 	auto cmp = [&max_diff] (int a, int b) -> bool {return max_diff[a] > max_diff[b];};
			// 	sort(order.begin(), order.end(), cmp);

			// 	for(auto d : order) {
			// 		auto k = shrink_symmetry(p, d);
			// 		if(!k) continue;
			// 		have = 1; ++count;
			// 		if(count <= 1) for(int i = 0; i < n; ++i) p4build[p[i].i] = p[i];
			// 	}
			// }

			if (len(sym_part_v) == 0) {

			} else {

			}

			pca_part(p);
			for(int i = 0; i < n; ++i) p[i] = p4build[p[i].i];
#ifdef DEBUG
cout << "debug end." << endl;
// cout << n << endl;
// for(auto p : p) cout << p.d[0] << ' ' << p.d[1] << ' ' << p.d[2] << endl;
exit(0);
#endif

			build(vector<P>(p.begin(), p.begin() + n / 2));
			build(vector<P>(p.begin() + n / 2, p.end()));
		}

	}	;

	vector<int> work(int N, vector<vector<float>> pts) {
		assert(len(pts) <= N);
		vector<P> p;
		for (int i = 0; i < len(pts); ++i)
			p.append({{pts[i][0], pts[i][1], pts[i][2]}, i});

		for (int n = len(p); len(p) != N; )
			p.append(p[randint(0, n - 1)]);


		Builder b;
		b.build(p);
		return b.arrange;
	}

	vector<vector<int>> work_batch(int N, vector<vector<vector<float>>> pts) {
		vector<vector<int>> ans;
		for (auto& p : pts) ans.append(work(N, p));
		return ans;
	}

}


#ifndef NOBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	using namespace builder_cpp;
	m.doc() = "Build K-D Tree";
	m.def("arrange", work, py::arg("N"), py::arg("pts"));
}

#else

int main() {
	using namespace builder_cpp;

	// vector<vector<float>> pts = {
	// 	{0., 0., 0.},
	// 	{0., 1., 0.},
	// 	{0., 2., 0.},
	// 	{0., 0., 1.},
	// 	{0., 1., 1.},
	// 	{0., 2., 1.},
	// 	{0., 0., 2.},
	// 	{0., 1., 2.},
	// 	{0., 2., 2.},
		
	// };
	// auto x = work_batch(16, {pts, pts});
	// for(auto i : x[0]) cout << i << ' ';
	// cout << endl;
	// for(auto i : x[1]) cout << i << ' ';
	// cout << endl;
	int N;
	cin >> N;
	vector<vector<float>> pts;
	float aug[3] = {1, 1, 1}; //{1, 1.5, 0.66};
	for(int i = 0; i < N; ++i) {
		float a, b, c;
		// cin >> a >> b >> c;
		cin >> b >> c >> a;
		a *= aug[0];
		b *= aug[1];
		c *= aug[2];
		pts.append(vector<float>{a, b, c});
	}
	work(pts.size(), pts);
}

#endif