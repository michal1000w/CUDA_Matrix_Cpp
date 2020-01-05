/*
#ifndef __CUDA_MATRIX__
#define __CUDA_MATRIX__

#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

typedef long long yeet;
using std::vector;
using std::string;

using std::cout;
using std::endl;
using std::vector;
using std::string;


template <typename Y>
class CuMatrix;

//////////////////////////////////////////////////KERNELS//////////////////////////////////////////////////////

template <typename Y>
__global__
void setNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = b[i];
}

template <typename Y>
__global__
void csetNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = b;
}

template <typename Y>
__global__
void addKernel(Y* a, const Y* b) {
    int i = threadIdx.x;
    a[i] += b[i];
}

template <typename Y>
__global__
void addNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] += b[i];
}

template <typename Y>
__global__
void caddNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] += b;
}

template <typename Y>
__global__
void subtractNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] -= b[i];
}

template <typename Y>
__global__
void csubtractNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] -= b;
}

template <typename Y>
__global__
void multiplyNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] *= b[i];
}

template <typename Y>
__global__
void cmultiplyNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] *= b;
}

template <typename Y>
__global__
void cdivideNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] /= b;
}

///////////////
template <typename Y>
__global__
void sigmoidKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = 1.0 / (1.0 + std::exp(-1 * a[i]));
}

template <typename Y>
__global__
void sigmoid_derivativeKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = a[i] * (1 - a[i]);
}

template <typename Y>
__global__
void squareKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] *= a[i];
}

template <typename Y>
__global__
void dotKernel(Y* a, Y* b, Y* c, yeet m, yeet n, yeet k) {
    yeet row = blockIdx.y * blockDim.y + threadIdx.y;
    yeet col = blockIdx.x * blockDim.x + threadIdx.x;
    Y sum = 0;
    if (col < k && row < m) {
        for (yeet i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

template <typename Y>
__global__
void transposeKernel(Y* mat_in, Y* mat_out, yeet rows, yeet cols) {
    yeet idx = blockIdx.x * blockDim.x + threadIdx.x;
    yeet idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        yeet pos = idy * cols + idx;
        yeet trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


//////////////////////////////////////////////////CLASS////////////////////////////////////////////////////////
template<typename Y>
class CuMatrix {
public:
    typedef Y value_type;
    CuMatrix() : _cols(0), _rows(0), _data(new Y[0]) {};
    CuMatrix(yeet rows, yeet cols);
    //new
    CuMatrix(yeet, yeet, mutable Y*);
    CuMatrix(yeet, yeet, vector<Y>);
    CuMatrix(const std::string&);
    CuMatrix(const CuMatrix<Y>&);

    bool exists(yeet row, yeet col) const;
    Y& operator()(yeet row, yeet col);
    Y operator()(yeet row, yeet col) const;
    CuMatrix<Y> operator=(const CuMatrix<Y>&);
    virtual ~CuMatrix();

    //friend void dotKernel(const CuMatrix<Y>* A, const CuMatrix<Y>* B, CuMatrix<Y>* C);


    /////NOWE
    CuMatrix<Y> operator+=(const CuMatrix<Y>&);
    CuMatrix<Y> operator+=(const Y&);
    CuMatrix<Y> operator-=(const CuMatrix<Y>&);
    CuMatrix<Y> operator-=(const Y&);
    CuMatrix<Y> operator*=(const CuMatrix<Y>&);
    CuMatrix<Y> operator*=(const Y&);
    CuMatrix<Y> operator/=(const Y&);

    //Dodatkowe
    CuMatrix<Y> operator+(const CuMatrix<Y>&);
    CuMatrix<Y> operator+(const Y&);
    CuMatrix<Y> operator-(const CuMatrix<Y>&);
    CuMatrix<Y> operator-(const Y&);
    CuMatrix<Y> operator*(const CuMatrix<Y>&);
    CuMatrix<Y> operator*(const Y&);
    CuMatrix<Y> operator/(const Y&);

    CuMatrix<Y> dot(const CuMatrix<Y>&);
    CuMatrix<Y> T();

    CuMatrix<Y> sigmoid();
    CuMatrix<Y> sigmoid_derivative();

    Y mean(); //for now works only on CPU //todo gpu compute
    CuMatrix<Y> square();


    //friend inline CuMatrix<Y>& sigmoid(const CuMatrix<Y>& m);
    //friend inline CuMatrix<Y>& sigmoid_derivative(const CuMatrix<Y>& m);

    CuMatrix<Y> print(short round = (short)5);
    void add(std::string);
    std::string getString(bool is_int = false);

    //////////

    yeet size() const { return _rows * _cols; }
    yeet rows() const { return _rows; }
    yeet cols() const { return _cols; }

    Y* getArray() const {
        Y* temp = new Y[size()];
        for (yeet i = 0; i < size(); i++)
            temp[i] = _data[i];
        return temp;
    }
    void print_size() { std::cout << "size : " << _rows << " x " << _cols << std::endl; }

    //setters
    void set_threads_per_block(unsigned int threads = 32) { this->THREADS_PER_BLOCK = threads; }
    void set_threads(unsigned int threads = 512) { this->THREADS = threads; }
private:
    unsigned power(short inp) {
        unsigned int output = 1;
        for (short i = 0; i < inp; i++)
            output *= 10;
        return output;
    }
    void initCuMatrix() {
        cudaMallocManaged(&this->_data, size() * sizeof(Y));
        //for (yeet i = 0; i < size(); i++) this->_data[i] = 0.0; //to przerobiæ na Cuda.set
        Cuda_set(this->_data, 0.0, size());
    }
    yeet _rows, _cols;
    mutable Y* _data;

    void Cuda_set(Y* a, Y* b, yeet size);
    void Cuda_set(Y* a, Y b, yeet size);
    unsigned int THREADS_PER_BLOCK = 32;
    unsigned int THREADS = 512;
public:
    void free();
};

//////////////////////////////////////////////////KONSTRUKTORY///////////////////////////////////////////////////////////////
template<typename Y>
CuMatrix<Y>::CuMatrix(yeet rows, yeet cols) : _rows(rows), _cols(cols) {
    this->initCuMatrix();
}

template <typename Y>
CuMatrix<Y>::CuMatrix(yeet rows, yeet cols, vector<Y> data) : _rows(rows), _cols(cols) {
    this->initCuMatrix();
    for (yeet i = 0; i < data.size(); i++)
        _data[i] = data[i];
}

template<typename Y>
CuMatrix<Y>::CuMatrix(yeet rows, yeet cols, mutable Y* _data) : _rows(rows), _cols(cols) {
    this->initCuMatrix();
    for (yeet i = 0; i < size(); i++) this->_data[i] = _data[i];
    Cuda_set(this->_data, _data, size());
}
template <typename Y>
CuMatrix<Y>::CuMatrix(const CuMatrix<Y>& mat) {
    this->_cols = mat._cols;
    this->_rows = mat._rows;
    this->initCuMatrix();
    Cuda_set(this->_data, mat._data, this->size());
}
template<typename Y>
CuMatrix<Y>::CuMatrix(const std::string& macierz) {
    //Podzia³ na fragmenty
    yeet len = macierz.length();
    std::string fragment = "";
    vector <std::string> fragmenty;

    for (yeet i = 0; i < len; i++) {
        if (macierz[i] == '[') {
            fragment = "";
            do {
                i++;
                if (macierz[i] == ']') break;

                fragment += macierz[i];
            } while (i < len - 1);
            fragmenty.push_back(fragment);
        }
    }

    this->_rows = fragmenty.size();

    //Podzia³ fragmentów na pojedyncze elementy
    vector <double> wartosci;
    short elementy = 0;

    for (yeet i = 0; i < this->_rows; i++) {
        len = fragmenty[i].size(); //d³ugoœæ stringa

        for (yeet j = 0; j < len; j++) {
            fragment = "";
            while (fragmenty[i][j] != ',' && j < len) {
                fragment += fragmenty[i][j];
                j++;
            }
            string::size_type sz;
            wartosci.push_back(stod(fragment, &sz));
        }

        if (i == 0) elementy = wartosci.size();
    }

    this->_cols = elementy;

    //Inicjowanie nowej macierzy typu CuMatrix
    this->initCuMatrix();

    //Przenoszenie elementów z vector do macierzy typu CuMatrix
    yeet i = 0;
    for (yeet y = 0; y < this->_rows; y++) {
        for (yeet x = 0; x < this->_cols; x++) {
            this->_data[y * this->_cols + x] = wartosci[i];
            i++;
        }
    }

    //czyszczenie pamiêci
    fragmenty.clear();
    wartosci.clear();
}

//////////////////////////////////////////////////DESTRUKTOR///////////////////////////////////////////////////////////////
template<typename Y>
CuMatrix<Y>::~CuMatrix() {
    this->free(); //tutaj
}
//////////////////////////////////////////////////DODATKOWE///////////////////////////////////////////////////////////////
template<typename Y>
void CuMatrix<Y>::free() {
    cudaFree(_data);
}

template<typename Y>
CuMatrix<Y> CuMatrix<Y>::print(short roundness) {
    int pomocnicza = 0;
    roundness = (roundness < 5 ? roundness : 4);
    for (yeet i = 0; i < this->_rows; i++) {
        std::cout << "[";
        for (yeet j = 0; j < this->_cols; j++) {
            if (roundness != 0) {
                pomocnicza = (float)this->_data[i * this->_cols + j] * power(roundness);
                std::cout << " " << ((float)pomocnicza / power(roundness));
            }
            else
                std::cout << " " << round(this->_data[i * this->_cols + j]);
        }
        std::cout << " ]" << std::endl;
    }
    return *this;
}

template <typename Y>
std::string CuMatrix<Y>::getString(bool is_int) {
    std::string data = "";
    CuMatrix<Y> temp(*this);
    for (int i = 0; i < this->_rows; i++) {
        data += "[";
        for (int j = 0; j < this->_cols; j++) {
            if (!(is_int)) data += to_string(temp(i, j));
            else data += to_string(int(temp(i, j)));

            if (j < this->_cols - 1) data += ",";
        }
        data += "]";
    }
    return data;
}

template<typename Y>
bool CuMatrix<Y>::exists(yeet row, yeet col) const {
    return (row < _rows && col < _cols);
}

template <typename Y>
void CuMatrix<Y>::add(std::string macierz) {
    //Podzia³ na fragmenty
    yeet len = macierz.length();
    std::string fragment = "";
    vector <std::string> fragmenty;

    for (yeet i = 0; i < len; i++) {
        if (macierz[i] == '[') {
            fragment = "";
            do {
                i++;
                if (macierz[i] == ']') break;

                fragment += macierz[i];
            } while (i < len - 1);
            fragmenty.push_back(fragment);
        }
    }

    this->_rows = fragmenty.size();

    //Podzia³ fragmentów na pojedyncze elementy
    vector <double> wartosci;
    short elementy = 0;

    for (yeet i = 0; i < this->_rows; i++) {
        len = fragmenty[i].size(); //d³ugoœæ stringa

        for (yeet j = 0; j < len; j++) {
            fragment = "";
            while (fragmenty[i][j] != ',' && j < len) {
                fragment += fragmenty[i][j];
                j++;
            }
            string::size_type sz;
            wartosci.push_back(stod(fragment, &sz));
        }

        if (i == 0) elementy = wartosci.size();
    }

    this->_cols = elementy;

    //czyszczenie starej macierzy
    this->free();

    //Inicjowanie nowej macierzy typu CuMatrix
    this->initCuMatrix();

    //Przenoszenie elementów z vector do macierzy typu CuMatrix
    yeet i = 0;
    for (yeet y = 0; y < this->_rows; y++) {
        for (yeet x = 0; x < this->_cols; x++) {
            this->_data[y * this->_cols + x] = wartosci[i];
            i++;
        }
    }

    //czyszczenie pamiêci
    fragmenty.clear();
    wartosci.clear();
}

template <typename Y>
void CuMatrix<Y>::Cuda_set(Y* a, Y* b, yeet size) {
    setNKernel << < _cols, _rows >> > (a, b, size);
    cudaDeviceSynchronize();
}

template <typename Y>
void CuMatrix<Y>::Cuda_set(Y* a, Y b, yeet size) {
    csetNKernel << < _cols, _rows >> > (a, b, size);
    cudaDeviceSynchronize();
}


///////////////////////////////////////////////////////////////OPERATORY//////////////////////////////////////////////////////////////
/////////////////Operatory//////////////////////////////////////
template <typename Y>
Y& CuMatrix<Y>::operator()(yeet row, yeet col) {
    return _data[_cols * row + col];
}

template <typename Y>
Y CuMatrix<Y>::operator()(yeet row, yeet col) const {
    return _data[_cols * row + col];
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator=(const CuMatrix<Y>& rhs) {
    this->free();

    this->_rows = rhs._rows;
    this->_cols = rhs._cols;

    this->initCuMatrix();
    Cuda_set(this->_data, rhs._data, this->size());

    return *this;
}


template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+=(const CuMatrix<Y>& rhs) {
    if (_rows <= THREADS)
        addNKernel << <_cols, _rows >> > (this->_data, rhs._data, size());
    else
        addNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs._data, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+=(const Y& rhs) {
    if (_rows <= THREADS)
        caddNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        caddNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-=(const CuMatrix<Y>& rhs) {
    if (_rows <= THREADS)
        subtractNKernel << <_cols, _rows >> > (this->_data, rhs._data, size());
    else
        subtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs._data, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-=(const Y& rhs) {
    if (_rows <= THREADS)
        csubtractNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        csubtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*=(const CuMatrix<Y>& rhs) {
    if (_rows <= THREADS)
        multiplyNKernel << <_cols, _rows >> > (this->_data, rhs._data, size());
    else
        multiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs._data, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*=(const Y& rhs) {
    if (_rows <= THREADS)
        cmultiplyNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        cmultiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator/=(const Y& rhs) {
    if (_rows <= THREADS)
        cdivideNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        cdivideNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}


///////////////
template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        addNKernel << <_cols, _rows >> > (temp._data, rhs._data, size());
    else
        addNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+(const Y& rhs) {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        caddNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        caddNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        subtractNKernel << <_cols, _rows >> > (temp._data, rhs._data, size());
    else
        subtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-(const Y& rhs) {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        csubtractNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        csubtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        multiplyNKernel << <_cols, _rows >> > (temp._data, rhs._data, size());
    else
        multiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*(const Y& rhs) {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        cmultiplyNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        cmultiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator/(const Y& rhs) {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        cdivideNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        cdivideNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}



////////////////////////////MATEMATYCZNE///////////////////////////////
template <typename Y>
CuMatrix<Y> CuMatrix<Y>::sigmoid() {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        sigmoidKernel << <_cols, _rows >> > (temp._data, size());
    else
        sigmoidKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::sigmoid_derivative() {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        sigmoid_derivativeKernel << <_cols, _rows >> > (temp._data, size());
    else
        sigmoid_derivativeKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::square() {
    CuMatrix<Y> temp(*this);
    if (_rows <= THREADS)
        squareKernel << <_cols, _rows >> > (temp._data, size());
    else
        squareKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::dot(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(this->_rows, rhs._cols);

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((rhs._cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (this->_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dotKernel << <grid, threads >> > (this->_data, rhs._data, temp._data, this->_rows, this->_cols, rhs._cols);
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::T() {
    CuMatrix<Y> temp(this->_cols, this->_rows);

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((size() - 1) / THREADS_PER_BLOCK + 1, (size() - 1) / THREADS_PER_BLOCK + 1);
    transposeKernel << <grid, threads >> > (this->_data, temp._data, this->_rows, this->_cols);
    cudaDeviceSynchronize();
    return temp;
}


template <typename Y>
Y CuMatrix<Y>::mean() {   ///this works on CPU only //todo
    Y output = 0.0;
    Y count = this->size();
    Y suma = 0.0;

    Y* arr = this->getArray();

    for (yeet i = 0; i < count; i++)
        suma += arr[i];

    output = (suma / (float)count);

    delete[] arr;

    return output;
}
#endif
*/

////////////////////////////OLD 2///////////////////////////////////
/*
#ifndef __CUDA_MATRIX__
#define __CUDA_MATRIX__

#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

typedef long long yeet;
using std::vector;
using std::string;

using std::cout;
using std::endl;
using std::vector;
using std::string;


template <typename Y>
class CuMatrix;

//////////////////////////////////////////////////KERNELS//////////////////////////////////////////////////////

template <typename Y>
__global__
void setNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = b[i];
}

template <typename Y>
__global__
void csetNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = b;
}

template <typename Y>
__global__
void addKernel(Y* a, const Y* b) {
    int i = threadIdx.x;
    a[i] += b[i];
}

template <typename Y>
__global__
void addNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] += b[i];
}

template <typename Y>
__global__
void caddNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] += b;
}

template <typename Y>
__global__
void subtractNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] -= b[i];
}

template <typename Y>
__global__
void csubtractNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] -= b;
}

template <typename Y>
__global__
void multiplyNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] *= b[i];
}

template <typename Y>
__global__
void cmultiplyNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] *= b;
}

template <typename Y>
__global__
void cdivideNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] /= b;
}

///////////////
template <typename Y>
__global__
void sigmoidKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = 1.0 / (1.0 + std::exp(-1 * a[i]));
}

template <typename Y>
__global__
void sigmoid_derivativeKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] = a[i] * (1 - a[i]);
}

template <typename Y>
__global__
void squareKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (i; i < n; i += blockDim.x * gridDim.x)
        a[i] *= a[i];
}

template <typename Y>
__global__
void dotKernel(Y* a, Y* b, Y* c, yeet m, yeet n, yeet k) {
    yeet row = blockIdx.y * blockDim.y + threadIdx.y;
    yeet col = blockIdx.x * blockDim.x + threadIdx.x;
    Y sum = 0;
    if (col < k && row < m) {
        for (yeet i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

template <typename Y>
__global__
void transposeKernel(Y* mat_in, Y* mat_out, yeet rows, yeet cols) {
    yeet idx = blockIdx.x * blockDim.x + threadIdx.x;
    yeet idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        yeet pos = idy * cols + idx;
        yeet trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


//////////////////////////////////////////////////CLASS////////////////////////////////////////////////////////
template<typename Y>
class CuMatrix {
public:
    typedef Y value_type;
    CuMatrix() : _cols(0), _rows(0), _data(new Y[0]) {};
    CuMatrix(yeet rows, yeet cols);
    //new
    CuMatrix(yeet, yeet,  Y*);
    CuMatrix(yeet, yeet, vector<Y>);
    CuMatrix(const std::string&);
    CuMatrix(const CuMatrix<Y>&);

    bool exists(yeet row, yeet col) const;
    Y& operator()(yeet row, yeet col);
    Y operator()(yeet row, yeet col) const;
    CuMatrix<Y> operator=(const CuMatrix<Y>&);
    virtual ~CuMatrix();

    //friend void dotKernel(const CuMatrix<Y>* A, const CuMatrix<Y>* B, CuMatrix<Y>* C);


    /////NOWE
    CuMatrix<Y> operator+=(const CuMatrix<Y>&);
    CuMatrix<Y> operator+=(const Y&);
    CuMatrix<Y> operator-=(const CuMatrix<Y>&);
    CuMatrix<Y> operator-=(const Y&);
    CuMatrix<Y> operator*=(const CuMatrix<Y>&);
    CuMatrix<Y> operator*=(const Y&);
    CuMatrix<Y> operator/=(const Y&);

    //Dodatkowe
    CuMatrix<Y> operator+(const CuMatrix<Y>&);
    CuMatrix<Y> operator+(const Y&);
    CuMatrix<Y> operator-(const CuMatrix<Y>&);
    CuMatrix<Y> operator-(const Y&);
    CuMatrix<Y> operator*(const CuMatrix<Y>&);
    CuMatrix<Y> operator*(const Y&);
    CuMatrix<Y> operator/(const Y&);

    CuMatrix<Y> dot(const CuMatrix<Y>&);
    CuMatrix<Y> T();

    CuMatrix<Y> sigmoid();
    CuMatrix<Y> sigmoid_derivative();

    Y mean(); //for now works only on CPU //todo gpu compute
    CuMatrix<Y> square();


    //friend inline CuMatrix<Y>& sigmoid(const CuMatrix<Y>& m);
    //friend inline CuMatrix<Y>& sigmoid_derivative(const CuMatrix<Y>& m);

    CuMatrix<Y> print(short round = (short)5);
    void add(std::string);
    std::string getString(bool is_int = false);

    //////////

    yeet size() const { return _rows * _cols; }
    yeet rows() const { return _rows; }
    yeet cols() const { return _cols; }

    Y* getArray() const {
        Y* temp = new Y[size()];
        for (yeet i = 0; i < size(); i++)
            temp[i] = _data[i];
        return temp;
    }
    void print_size() { std::cout << "size : " << _rows << " x " << _cols << std::endl; }

    //setters
    void set_threads_per_block(unsigned int threads = 32) { this->THREADS_PER_BLOCK = threads; }
    void set_threads(unsigned int threads = 512) { this->THREADS = threads; }
private:
    unsigned power(short inp) {
        unsigned int output = 1;
        for (short i = 0; i < inp; i++)
            output *= 10;
        return output;
    }
    void initCuMatrix() {
        cudaMallocManaged(&this->_data, size() * sizeof(Y));
        cudaSetDevice(0);

        //prefetch for faster computation
        int device = -1;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(this->_data, size() * sizeof(Y), device, NULL);

        Cuda_set(this->_data, 0.0, size());
    }
    yeet _rows, _cols;
    Y* _data;

    void Cuda_set(Y* a, Y* b, yeet size);
    void Cuda_set(Y* a, Y b, yeet size);
    unsigned int THREADS_PER_BLOCK = 32;
    unsigned int THREADS = 256;
public:
    void free();
};

//////////////////////////////////////////////////KONSTRUKTORY///////////////////////////////////////////////////////////////
template<typename Y>
CuMatrix<Y>::CuMatrix(yeet rows, yeet cols) : _rows(rows), _cols(cols) {
    this->initCuMatrix();
}

template <typename Y>
CuMatrix<Y>::CuMatrix(yeet rows, yeet cols, vector<Y> data) : _rows(rows), _cols(cols) {
    this->initCuMatrix();
    for (yeet i = 0; i < data.size(); i++)
        _data[i] = data[i];
}

template<typename Y>
CuMatrix<Y>::CuMatrix(yeet rows, yeet cols,  Y* _data) : _rows(rows), _cols(cols) {
    this->initCuMatrix();
    for (yeet i = 0; i < size(); i++) this->_data[i] = _data[i];
    Cuda_set(this->_data, _data, size());
}
template <typename Y>
CuMatrix<Y>::CuMatrix(const CuMatrix<Y>& mat) {
    this->_cols = mat._cols;
    this->_rows = mat._rows;
    this->initCuMatrix();
    Cuda_set(this->_data, mat._data, this->size());
}
template<typename Y>
CuMatrix<Y>::CuMatrix(const std::string& macierz) {
    //Podzia³ na fragmenty
    yeet len = macierz.length();
    std::string fragment = "";
    vector <std::string> fragmenty;

    for (yeet i = 0; i < len; i++) {
        if (macierz[i] == '[') {
            fragment = "";
            do {
                i++;
                if (macierz[i] == ']') break;

                fragment += macierz[i];
            } while (i < len - 1);
            fragmenty.push_back(fragment);
        }
    }

    this->_rows = fragmenty.size();

    //Podzia³ fragmentów na pojedyncze elementy
    vector <double> wartosci;
    short elementy = 0;

    for (yeet i = 0; i < this->_rows; i++) {
        len = fragmenty[i].size(); //d³ugoœæ stringa

        for (yeet j = 0; j < len; j++) {
            fragment = "";
            while (fragmenty[i][j] != ',' && j < len) {
                fragment += fragmenty[i][j];
                j++;
            }
            string::size_type sz;
            wartosci.push_back(stod(fragment, &sz));
        }

        if (i == 0) elementy = wartosci.size();
    }

    this->_cols = elementy;

    //Inicjowanie nowej macierzy typu CuMatrix
    this->initCuMatrix();

    //Przenoszenie elementów z vector do macierzy typu CuMatrix
    yeet i = 0;
    for (yeet y = 0; y < this->_rows; y++) {
        for (yeet x = 0; x < this->_cols; x++) {
            this->_data[y * this->_cols + x] = wartosci[i];
            i++;
        }
    }

    //czyszczenie pamiêci
    fragmenty.clear();
    wartosci.clear();
}

//////////////////////////////////////////////////DESTRUKTOR///////////////////////////////////////////////////////////////
template<typename Y>
CuMatrix<Y>::~CuMatrix() {
    this->free(); //tutaj
}
//////////////////////////////////////////////////DODATKOWE///////////////////////////////////////////////////////////////
template<typename Y>
void CuMatrix<Y>::free() {
    cudaFree(_data);
}

template<typename Y>
CuMatrix<Y> CuMatrix<Y>::print(short roundness) {
    int pomocnicza = 0;
    roundness = (roundness < 5 ? roundness : 4);
    for (yeet i = 0; i < this->_rows; i++) {
        std::cout << "[";
        for (yeet j = 0; j < this->_cols; j++) {
            if (roundness != 0) {
                pomocnicza = (float)this->_data[i * this->_cols + j] * power(roundness);
                std::cout << " " << ((float)pomocnicza / power(roundness));
            }
            else
                std::cout << " " << round(this->_data[i * this->_cols + j]);
        }
        std::cout << " ]" << std::endl;
    }
    return *this;
}

template <typename Y>
std::string CuMatrix<Y>::getString(bool is_int) {
    std::string data = "";
    CuMatrix<Y> temp(*this);
    for (int i = 0; i < this->_rows; i++) {
        data += "[";
        for (int j = 0; j < this->_cols; j++) {
            if (!(is_int)) data += to_string(temp(i, j));
            else data += to_string(int(temp(i, j)));

            if (j < this->_cols - 1) data += ",";
        }
        data += "]";
    }
    return data;
}

template<typename Y>
bool CuMatrix<Y>::exists(yeet row, yeet col) const {
    return (row < _rows && col < _cols);
}

template <typename Y>
void CuMatrix<Y>::add(std::string macierz) {
    //Podzia³ na fragmenty
    yeet len = macierz.length();
    std::string fragment = "";
    vector <std::string> fragmenty;

    for (yeet i = 0; i < len; i++) {
        if (macierz[i] == '[') {
            fragment = "";
            do {
                i++;
                if (macierz[i] == ']') break;

                fragment += macierz[i];
            } while (i < len - 1);
            fragmenty.push_back(fragment);
        }
    }

    this->_rows = fragmenty.size();

    //Podzia³ fragmentów na pojedyncze elementy
    vector <double> wartosci;
    short elementy = 0;

    for (yeet i = 0; i < this->_rows; i++) {
        len = fragmenty[i].size(); //d³ugoœæ stringa

        for (yeet j = 0; j < len; j++) {
            fragment = "";
            while (fragmenty[i][j] != ',' && j < len) {
                fragment += fragmenty[i][j];
                j++;
            }
            string::size_type sz;
            wartosci.push_back(stod(fragment, &sz));
        }

        if (i == 0) elementy = wartosci.size();
    }

    this->_cols = elementy;

    //czyszczenie starej macierzy
    this->free();

    //Inicjowanie nowej macierzy typu CuMatrix
    this->initCuMatrix();

    //Przenoszenie elementów z vector do macierzy typu CuMatrix
    yeet i = 0;
    for (yeet y = 0; y < this->_rows; y++) {
        for (yeet x = 0; x < this->_cols; x++) {
            this->_data[y * this->_cols + x] = wartosci[i];
            i++;
        }
    }

    //czyszczenie pamiêci
    fragmenty.clear();
    wartosci.clear();
}

template <typename Y>
void CuMatrix<Y>::Cuda_set(Y* a, Y* b, yeet size) {
    if (_rows <= THREADS)
        setNKernel << <_cols, _rows >> > (a, b, size);
    else
        setNKernel << < (size + THREADS - 1) / THREADS, THREADS >> > (a, b, size);
    cudaDeviceSynchronize();
}

template <typename Y>
void CuMatrix<Y>::Cuda_set(Y* a, Y b, yeet size) {
    if (_rows <= THREADS)
        csetNKernel << <_cols, _rows >> > (a, b, size);
    else
        csetNKernel << <(size + THREADS - 1) / THREADS, THREADS >> > (a, b, size);
    cudaDeviceSynchronize();
}


///////////////////////////////////////////////////////////////OPERATORY//////////////////////////////////////////////////////////////
/////////////////Operatory//////////////////////////////////////
template <typename Y>
Y& CuMatrix<Y>::operator()(yeet row, yeet col) {
    return _data[_cols * row + col];
}

template <typename Y>
Y CuMatrix<Y>::operator()(yeet row, yeet col) const {
    return _data[_cols * row + col];
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator=(const CuMatrix<Y>& rhs) {
    this->free();

    this->_rows = rhs._rows;
    this->_cols = rhs._cols;

    this->initCuMatrix();
    Cuda_set(this->_data, rhs._data, this->size());

    return *this;
}


template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+=(const CuMatrix<Y>& rhs) {
    if (_rows <= THREADS)
        addNKernel << <_cols, _rows >> > (this->_data, rhs._data, size());
    else
        addNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs._data, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+=(const Y& rhs) {
    if (_rows <= THREADS)
        caddNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        caddNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-=(const CuMatrix<Y>& rhs) {
    if (_rows <= THREADS)
        subtractNKernel << <_cols, _rows >> > (this->_data, rhs._data, size());
    else
        subtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs._data, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-=(const Y& rhs) {
    if (_rows <= THREADS)
        csubtractNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        csubtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*=(const CuMatrix<Y>& rhs) {
    if (_rows <= THREADS)
        multiplyNKernel << <_cols, _rows >> > (this->_data, rhs._data, size());
    else
        multiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs._data, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*=(const Y& rhs) {
    if (_rows <= THREADS)
        cmultiplyNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        cmultiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator/=(const Y& rhs) {
    if (_rows <= THREADS)
        cdivideNKernel << <_cols, _rows >> > (this->_data, rhs, size());
    else
        cdivideNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (this->_data, rhs, size());
    cudaDeviceSynchronize();
    return *this;
}


///////////////
template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        addNKernel << <_cols, _rows >> > (temp._data, rhs._data, size());
    else
        addNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator+(const Y& rhs) {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        caddNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        caddNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        subtractNKernel << <_cols, _rows >> > (temp._data, rhs._data, size());
    else
        subtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator-(const Y& rhs) {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        csubtractNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        csubtractNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        multiplyNKernel << <_cols, _rows >> > (temp._data, rhs._data, size());
    else
        multiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator*(const Y& rhs) {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        cmultiplyNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        cmultiplyNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::operator/(const Y& rhs) {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        cdivideNKernel << <_cols, _rows >> > (temp._data, rhs, size());
    else
        cdivideNKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, rhs, size());
    cudaDeviceSynchronize();
    return temp;
}



////////////////////////////MATEMATYCZNE///////////////////////////////
template <typename Y>
CuMatrix<Y> CuMatrix<Y>::sigmoid() {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        sigmoidKernel << <_cols, _rows >> > (temp._data, size());
    else
        sigmoidKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::sigmoid_derivative() {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        sigmoid_derivativeKernel << <_cols, _rows >> > (temp._data, size());
    else
        sigmoid_derivativeKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::square() {
    CuMatrix<Y> temp(*this);

    if (_rows <= THREADS)
        squareKernel << <_cols, _rows >> > (temp._data, size());
    else
        squareKernel << <(size() + THREADS - 1) / THREADS, THREADS >> > (temp._data, size());
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::dot(const CuMatrix<Y>& rhs) {
    CuMatrix<Y> temp(this->_rows, rhs._cols);

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((rhs._cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (this->_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dotKernel << <grid, threads >> > (this->_data, rhs._data, temp._data, this->_rows, this->_cols, rhs._cols);
    cudaDeviceSynchronize();
    return temp;
}

template <typename Y>
CuMatrix<Y> CuMatrix<Y>::T() {
    CuMatrix<Y> temp(this->_cols, this->_rows);

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((size() - 1) / THREADS_PER_BLOCK + 1, (size() - 1) / THREADS_PER_BLOCK + 1);
    transposeKernel << <grid, threads >> > (this->_data, temp._data, this->_rows, this->_cols);
    cudaDeviceSynchronize();
    return temp;
}


template <typename Y>
Y CuMatrix<Y>::mean() {   ///this works on CPU only //todo
    Y output = 0.0;
    Y count = this->size();
    Y suma = 0.0;

    Y* arr = this->getArray();

    for (yeet i = 0; i < count; i++)
        suma += arr[i];

    output = (suma / (float)count);

    delete[] arr;

    return output;
}
#endif
*/