#include <iostream>
#include <map>
#include <fstream>

#include <boost/numeric/odeint.hpp>

typedef boost::numeric::ublas::matrix<double> state_type;

using namespace std;

class write_snapshots
{
public:
	typedef std::map<size_t, std::string> map_type;

	write_snapshots(void) : m_count(0) {}

	void operator()(const state_type &x, double t)
	{
		map<size_t, string>::const_iterator it = m_snapshots.find(m_count);
		if (it != m_snapshots.end())
		{
			ofstream fout(it->second.c_str());
			fout << x.size1() << "\t" << x.size2() << "\n";
			for (size_t i = 0; i < x.size1(); ++i)
			{
				for (size_t j = 0; j < x.size2(); ++j)
				{
					fout << i << "\t" << j << "\t" << x(i, j) << "\n";
				}
			}
		}
		++m_count;
	}

	map_type &snapshots(void) { return m_snapshots; }
	const map_type &snapshots(void) const { return m_snapshots; }

private:
	size_t m_count;
	map_type m_snapshots;
};
