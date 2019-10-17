#include <data/matrix.hpp>

typedef matrix<double> state_type;

namespace boost
{

namespace numeric
{
namespace odeint
{

template <>
struct is_resizeable<state_type>
{ // declare resizeability
    const static bool value = boost::false_type::value;
};

template <>
struct same_size_impl<state_type, state_type>
{ // define how to check size
    static bool same_size(const state_type &v1,
                          const state_type &v2)
    {
        return v1.size() == v2.size();
    }
};

template <>
struct resize_impl<state_type, state_type>
{ // define how to resize
    static void resize(state_type &v1,
                       const state_type &v2)
    {
        // v1.resize(v2.size());
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost