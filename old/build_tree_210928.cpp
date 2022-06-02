#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <algorithm>
#include <cassert>

namespace immortalCO {
	const int MaxNode = 1000010;
	const int MaxL = 20;
	const int seed = 674433238;
	bool structure;
	int sample;
	
	#define len(x) ((int) (x).size())

	int cur_D;
	struct Point {
		double d[3];
		int i;

		bool operator < (const Point& p) const {
			return d[cur_D] < p.d[cur_D];
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

	int build(std::vector<Point> pts, int dep) {
		assert(__builtin_popcount(pts.size()) == 1);

		int p = tot++;
		layers[dep].push_back(p);

		if(len(pts) == 1) {
			tree[p].pid = pts[0].i;
			tree[p].l = -1;
			tree[p].r = -1;
			tree[p].s = -1;
		} else {
			tree[p].pid = -1;

			int m = pts.size() >> 1;
			cur_D = 2 - dep % 3; // z, y, x, z, y, x, ...
			std::nth_element(pts.begin(), pts.begin() + m, pts.end());

			tree[p].l = build(std::vector<Point>(pts.begin(), pts.begin() + m), dep + 1);
			tree[p].r = build(std::vector<Point>(pts.begin() + m, pts.end()), dep + 1);

			int spl = pts.size() >> sample;

			if(!spl) tree[p].s = -1;
			else {
				std::shuffle(pts.begin(), pts.end(), std::default_random_engine(rand()));
				tree[p].s = build(std::vector<Point>(pts.begin(), pts.begin() + spl), dep + sample);
			}
		}
		return p;
	}
}

int main(int argc, char** argv) {
	using namespace immortalCO;
	srand(seed);

	sample = atoi(argv[1]);
	structure = argc > 2 && std::string(argv[2]) == "struct";

	if(!structure || argc <= 3) {
		for(double x, y, z; scanf("%lf%lf%lf", &x, &y, &z) != -1; ) {
			x += (rand() % 2000 - 1000) / 1000 * eps;
			y += (rand() % 2000 - 1000) / 1000 * eps;
			z += (rand() % 2000 - 1000) / 1000 * eps;
			pts.push_back({{x, y, z}, (int) pts.size()});
		}
	} else {
		int N = std::atoi(argv[3]);
		pts.resize(N);
	}


	int input_N = 0;
	if(argc > 3) input_N = std::atoi(argv[3]);

	while(__builtin_popcount(pts.size()) != 1 || len(pts) != input_N)
		pts.push_back(pts[rand() % pts.size()]);
	
	build(pts, 0);

	if(structure) {
		printf("size %d\n", len(pts));
		printf("sample %d\n", sample);
	}

	for(int l = MaxL; l--; ) {
		if(layers[l].empty()) continue;
		if(structure) {
			printf("layer %d\n", l);
			for(int p : layers[l]) {
				auto& pp = tree[p];
				printf("node %d %d %d %d %d\n", p, pp.pid, pp.l, pp.r, pp.s);
			}
		} else {
			for(int p : layers[l]) {
				auto& pp = tree[p];
				printf("%d\n", pp.pid);
			}
			break;
		}
	}
}