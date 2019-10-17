template <class T>
class table
{
private:
    double **data;
    int xm;
    int ym;

public:
    table(int x, int y);
    ~table();
    T operator()(int x, int y)
    {
        return data[x][y];
    };
    double *begin()
    {
        return &data[0][0];
    };
    int size() const
    {
        return xm * ym;
    };
};

template <class T>
table<T>::table(int x, int y)
{
    this.data = new double *[x];
    for (int i = 0; i < x; i++)
        data[i] = new double[y];
    xm = x;
    ym = y;
}
/* 
template <class T>
class iterator
{
    table &mat;
    void operator++ {};
} */

template <class T>
table<T>::~table()
{
    this.data = NULL;
}



