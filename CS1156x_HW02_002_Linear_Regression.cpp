#include <vector>
#include <functional>
#include <random>
#include <chrono>

#include "armadillo"

using namespace std;
using namespace arma;

struct feature
{
	feature(int dim): d(dim) { val.resize(d, 0.0); }
	vector<double> val;
	int d;
	char cat;
};

template<class Func>
feature generateFeature(const int dim, Func randomCoord)
{
	feature p(dim);

	for (int i = 1; i < dim; i++)
	{
		p.val[i] = randomCoord();
	}

	p.val[0] = 1;

	return p;
}

template<class Func>
feature generateHypo(const int dim, Func randomCoord)
{
	feature res;
	feature p1(dim), p2(dim);

	p1 = generateFeature(dim, randomCoord);
	p2 = generateFeature(dim, randomCoord);

	res.val[0] = p2.val[2] - p1.val[2];
	res.val[1] = p1.val[1] - p2.val[1];
	res.val[2] = p1.val[1] * (p1.val[2] - p2.val[2]) + p1.val[2] * (p2.val[1] - p1.val[1]);

	return res;
}

char applyFunction(const feature& o, const feature& f)
{
	double det = 0;

	for (int i = 0; i < f.d; i++)
	{
		det += o.val[i] * f.val[i];
	}

	return (det < 0) ? -1 : 1;
}

char applyNonLinearFunction(const feature& o, const feature& f)
{
	double det = o.val[1] * o.val[1] + o.val[2] * o.val[2] - 0.6;

	return (det < 0) ? -1 : 1;
}

template<class Func1, class Func2>
void generateData(const int& D, const feature& f, vector<feature> &data, Func1 apply , Func2 randomCoord)
{
	data.clear();
	feature p(f.d);

	for (int i = 0; i < D; i++)
	{
		p = generateFeature(f.d, randomCoord);
		p.cat = apply(p, f);
		data.push_back(p);
	}
}

vector<feature> getMismatch(int N, const vector<feature>& data, const feature& f)
{
	vector<feature> res;
	int i = 0;

	for (vector<feature>::const_iterator it = data.begin(); it != data.end() && i < N; it++, i++)
	{
		float cat = applyFunction(*it, f);

		if (cat != (*it).cat)
		{
			res.push_back(*it);
		}
	}

	return res;
}

long runPLA(const int& N, const vector<feature>& data, feature& g)
{
	long it = 0;
	vector<feature> mismatch;

	for (mismatch = getMismatch(N, data, g); mismatch.size() > 0; mismatch = getMismatch(N, data, g))
	{
		it++;

		feature p = mismatch[rand() % mismatch.size()];

		for (int i = 0; i < g.d; i++)
		{
			g.val[i] += p.cat * p.val[i];
		}
	}

	return it;
}

void runLinearRegression(const int& N, const int& d, const vector<feature>& data, feature& g)
{
	mat x(N,d), xTrans, xDagger;
	mat y(N, 1);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < data[i].d; j++)
		{
			x(i, j) = data[i].val[j];
		}

		y(i, 0) = data[i].cat;
	}

	xTrans = trans(x);
	xDagger = inv(xTrans * x) * xTrans;

	mat w(1, d);

	w = xDagger * y;

	for (int i = 0; i < g.d; i++)
	{
		g.val[i] = w(i, 0);
	}
}

int main(int argc, char* argv[])
{
	const int N = 1000;
	const int D = 2000;
	const int d = 3;
	const int runs = 1000;
	long long sumPLA = 0;
	long long eIn = 0; 
	long long eOut = 0;
	double p = 0;

	feature f(d), g(d);
	feature sol(6);
	vector<feature> data, dataTrans;
	default_random_engine generator;
	generator.seed((unsigned long)chrono::system_clock::now().time_since_epoch().count());

	uniform_real_distribution<double> faceDistribution(-1.0, 1.0);
	auto randomCoord = bind(faceDistribution, ref(generator));

	uniform_int_distribution<int> noiseDistribution(1,10);
	auto randomNoise = bind(noiseDistribution, ref(generator));

	long ss[5] = { 0, 0, 0, 0, 0 };

	for (int i = 0; i < runs; i++)
	{
		//Questions 5-7
		//f = generateHypo(randomCoord);
		//generateData(D, f, data, applyFunction, randomCoord);

		//runLinearRegression(N, d, data, g);

		//for (vector<point>::const_iterator it = data.cbegin(); it != data.cbegin() + N; it++)
		//{
		//	char cat = applyFunction(*it, g);
		//	if (cat != (*it).cat)
		//	{
		//		eIn++;
		//	}
		//}

		//for (vector<point>::const_iterator it = data.cbegin() + N; it != data.cend(); it++)
		//{
		//	char cat = applyFunction(*it, g);
		//	if (cat != (*it).cat)
		//	{
		//		eOut++;
		//	}
		//}
		
		//sumPLA += runPLA(N, data, g);

		//Question 8
		//generateData(D, f, data, applyNonLinearFunction, randomCoord);
		//runLinearRegression(N, d, data, g);

		//for (vector<feature>::const_iterator it = data.cbegin(); it != data.cbegin() + N; it++)
		//{
		//	char cat = applyFunction(*it, g);
		//	if (cat != (*it).cat)
		//	{
		//		eIn++;
		//	}
		//}
	
		// Questions 9-10
		generateData(D, f, data, applyNonLinearFunction, randomCoord);
		for (vector<feature>::iterator it = data.begin(); it != data.end(); it++)
		{
			if (randomNoise() == 1)
			{
				(*it).cat = -(*it).cat;
			}
		}

		feature h(6);
		dataTrans.clear();
		
		for (vector<feature>::const_iterator it = data.cbegin(); it != data.cend(); it++)
		{
			feature p(6);
			feature a = *it;

			p.val[0] = 1.0;
			p.val[1] = a.val[1];
			p.val[2] = a.val[2];
			p.val[3] = a.val[1] * a.val[2];
			p.val[4] = a.val[1] * a.val[1];
			p.val[5] = a.val[2] * a.val[2];
			p.cat = a.cat;

			dataTrans.push_back(p);
		}

		runLinearRegression(N, 6, dataTrans, h);

		for (vector<feature>::const_iterator it = dataTrans.cbegin(); it != dataTrans.cbegin() + N; it++)
		{
			char cat = applyFunction(*it, h);
			if (cat != (*it).cat)
			{
				eIn++;
			}
		}

		for (vector<feature>::const_iterator it = dataTrans.cbegin() + N; it != dataTrans.cend(); it++)
		{
			char cat = applyFunction(*it, h);
			if (cat != (*it).cat)
			{
				eOut++;
			}
		}

		for (int j = 0; j < h.d; j++)
		{
			sol.val[j] += h.val[j];
		}

		feature g_a(6), g_b(6), g_c(6), g_d(6), g_e(6);

		g_a.val[0] = g_b.val[0] = g_c.val[0] = g_d.val[0] = g_e.val[0] = -1;
		g_a.val[1] = g_b.val[1] = g_c.val[1] = g_e.val[1] = -0.05; g_d.val[1] = -1.5;
		g_a.val[2] = g_b.val[2] = g_c.val[2] = g_d.val[2] = g_e.val[2] = 0.08;
		g_a.val[3] = g_b.val[3] = g_c.val[3] = g_d.val[3] = 0.13; g_e.val[3] = 1.5;
		g_a.val[4] = g_b.val[4] = 1.5;  g_c.val[4] = 15.0; g_d.val[4] = 0.05; g_e.val[4] = 0.15;
		g_a.val[5] = 1.5;  g_b.val[5] = 15.0; g_c.val[5] = 1.5; g_d.val[5] = 0.05; g_e.val[5] = 0.15;

		for (vector<feature>::const_iterator it = dataTrans.cbegin(); it != dataTrans.cbegin() + N; it++)
		{
			char catH = applyFunction(*it, h);
			char catA = applyFunction(*it, g_a);
			char catB = applyFunction(*it, g_b);
			char catC = applyFunction(*it, g_c);
			char catD = applyFunction(*it, g_d);
			char catE = applyFunction(*it, g_e);
			if (catH != catA)
			{
				ss[0]++;
			}
			if (catH != catB)
			{
				ss[1]++;
			}
			if (catH != catC)
			{
				ss[2]++;
			}
			if (catH != catD)
			{
				ss[3]++;
			}
			if (catH != catE)
			{
				ss[4]++;
			}
		}

		if (i % 10 == 0)
		{
			printf("Run %d completed!\n", i);
		}
	}

	//printf("For %d runs on N = %d we have %.3f iterations\n", runs, N, (float)sumPLA / runs);
	printf("For %d runs on N = %d we have E_in = %.4f and E_out = %.4f\n", runs, N, (float)eIn / runs / N, (float)eOut / runs / (D - N));
	printf("W[0] = %.4f  W[1] = %.4f  W[2] = %.4f  W[3] = %.4f  W[4] = %.4f  W[5] = %.4f\n", sol.val[0] / N, sol.val[1] / N, sol.val[2] / N, sol.val[3] / N, sol.val[4] / N, sol.val[5] / N );

	printf("ssA = %d  ssB = %d  ssC = %d  ssD = %d  ssE = %d\n", ss[0], ss[1], ss[2], ss[3], ss[4], ss[5]);

	return 0;
}

// Q9-10: ssA = 38299  ssB = 336793  ssC = 336671  ssD = 368158  ssE = 439403
// Q9-10: For 1000 runs on N = 1000 we have E_in = 0.1245 and E_out = 0.1267