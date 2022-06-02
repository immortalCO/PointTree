#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <algorithm>
#include <cassert>
#include <functional>
#include <cassert>
#include <cmath>
#define append push_back

namespace immortalCO {
	using namespace std;
	const int MaxNode = 1000010;
	const int MaxP = 16385;
	const int MaxL = 20;
	const int MaxV = 512;
	const int seed = 674433238;
	bool structure;
	int sample;
	
	#define len(x) ((int) (x).size())

	
	double v[MaxV][3], *cur_v;
	int num_v;

	struct Point {
		double d[3];
		int i;

		double mem_dist;
		void calc_dist() {
			mem_dist = d[0] * cur_v[0] + d[1] * cur_v[1] + d[2] * cur_v[2];
		}

		bool operator < (const Point& p) const {
			return mem_dist < p.mem_dist;
		}
	};
	std::vector<Point> pts;

	const double eps = 1e-8;

	struct Node {
		int pid;
		int l, r, s;

	}	tree[MaxNode];
	int tot;

	std::vector<int> layers[MaxL];

	struct Carry {
		int cho;
		double mov;
		double operator() (const Point& p) {
			if(cho == -1) return 0;
			auto cur_v = v[cho];
			auto d = p.d;
			return d[0] * cur_v[0] + d[1] * cur_v[1] + d[2] * cur_v[2] - mov;
		}
	};
	std::vector<double> extra[MaxP];

	void arrange(std::vector<Point> &pts, int vec=-1) {
		if(vec != -1) cur_v = v[vec];
		for(auto &p : pts) p.calc_dist();
		int m = len(pts) >> 1;
		std::nth_element(pts.begin(), pts.begin() + m, pts.end());
	}

	int build(std::vector<Point> pts, int dep, std::vector<Carry> carry, int fixed=-1) {
		assert(__builtin_popcount(pts.size()) == 1);

		int p = tot++;
		layers[dep].push_back(p);

		if(len(pts) == 1) {
			auto &pt = pts[0];
			tree[p].pid = pt.i;
			tree[p].l = -1;
			tree[p].r = -1;
			tree[p].s = -1;

			// printf("%d\n", len(carry));

			if(len(carry) == dep) {
				vector<double> feature;
				for(auto &c : carry) feature.push_back(c(pt));
				extra[pt.i] = feature;
			}
			
		} else {
			tree[p].pid = -1;

			int m = pts.size() >> 1;
			int cho = -1;
			double best = -1e10;
			for(int vec = 0; vec != num_v; ++vec) {
				cur_v = v[vec];
				if(fixed != -1) {
					double* fix_v = v[fixed];
					double dot = cur_v[0] * fix_v[0] + cur_v[1] * fix_v[1] + cur_v[2] * fix_v[2];
					if(dot < 0) dot = -dot;
					if(dot > eps) continue;
					// printf("otho %d %d\n", vec, fixed);
				}
				arrange(pts, vec);
				double ls1 = 0, ls2 = 0, rs1 = 0, rs2 = 0;
				for(int i = 0; i < (m << 1); ++i) {
					double dist = pts[i].mem_dist;
					(i < m ? ls1 : rs1) += dist;
					(i < m ? ls2 : rs2) += dist * dist;
				}
				ls1 /= m; ls2 /= m; rs1 /= m; rs2 /= m;
				// double val = (rs1 - ls1) * (rs1 - ls1) + (ls2 - ls1 * ls1) + (rs2 - rs1 * rs1)
				double val = ls2 + rs2 - 2 * ls1 * rs1;
				// printf("vec = %d val = %.4lf best = %.4lf\n", vec, val, best);
				if(val > best) {best = val; cho = vec;}
			}

			tree[p].pid = cho;
			arrange(pts, cho);

			auto lmax = max_element(pts.begin(), pts.begin() + m)->mem_dist;
			auto rmin = min_element(pts.begin() + m, pts.end())->mem_dist;
			assert(lmax <= rmin);
			auto mov = (lmax + rmin) / 2;
			carry.push_back({cho, mov});

			// for(auto &x : std::vector<Point>(pts.begin(), pts.begin() + m)) assert(carry.back()(x) <= 0);
			// for(auto &x : std::vector<Point>(pts.begin() + m, pts.end())) assert(carry.back()(x) >= 0);

			tree[p].l = build(std::vector<Point>(pts.begin(), pts.begin() + m), dep + 1, carry);
			tree[p].r = build(std::vector<Point>(pts.begin() + m, pts.end()), dep + 1, carry);

			int spl = pts.size() >> sample;

			if(!spl) tree[p].s = -1;
			else {
				std::shuffle(pts.begin(), pts.end(), std::default_random_engine(rand()));
				// Force to be otho
				// for(int i = sample - 1; i--; ) carry.push_back({-1, 0.0});
				tree[p].s = build(std::vector<Point>(pts.begin(), pts.begin() + spl), dep + sample, carry, cho);
			}
		}
		return p;
	}
}

int main(int argc, char** argv) {
	using namespace immortalCO;
	srand(seed);

	sample = atoi(argv[1]);
	structure = 0;

	if(!structure || argc <= 3) {
		assert(scanf("%d", &num_v) == 1);
		for(int i = 0; i < num_v; ++i) 
			scanf("%lf%lf%lf", v[i] + 0, v[i] + 1, v[i] + 2);

		for(double x, y, z; scanf("%lf%lf%lf", &x, &y, &z) != -1; ) {
			// x += (rand() % 2000 - 1000) / 1000 * eps;
			// y += (rand() % 2000 - 1000) / 1000 * eps;
			// z += (rand() % 2000 - 1000) / 1000 * eps;
			pts.push_back({{x, y, z}, (int) pts.size()});
		}
	} else {
		int N = std::atoi(argv[3]);
		pts.resize(N);
	}

	int num_P = len(pts);


	int input_N = 0;
	if(argc > 3) input_N = std::atoi(argv[3]);

	while(__builtin_popcount(pts.size()) != 1 || len(pts) != input_N)
		pts.push_back(pts[rand() % pts.size()]);
	
	build(pts, 0, std::vector<Carry>());

	if(structure) {
		printf("size %d\n", len(pts));
		printf("sample %d\n", sample);
	}

	for(int l = MaxL; l--; ) {
		if(layers[l].empty()) continue;
		for(int p : layers[l]) {
			auto& pp = tree[p];
			printf("%d ", pp.pid);
		}
		printf("\n");
	}

	for(int i = 0; i < num_P; ++i) {
		for(auto &b : extra[i]) printf("%.8lf ", b);
		printf("\n");
	}
}