#include "CUDA_Class.cuh";

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

typedef long long yeet;
using std::vector;
using std::string;

template <typename Y>
void deletep(Y&) {}
template <typename Y>
void deletep(Y*& ptr) {
    delete ptr;
    ptr = nullptr;
}
template <typename Y>
void deletea(Y*& ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
        ptr = nullptr;
    }
}
template<typename Y>
class Matrix {
public:
    typedef Y value_type;
    Matrix() : _cols(0), _rows(0), _data(new Y[0]) {};
    Matrix(yeet rows, yeet cols);
    //new
    Matrix(yeet, yeet, mutable Y*);
    Matrix(yeet, yeet, vector<Y>);
    Matrix(const std::string&);
    Matrix(const Matrix<Y>&);

    bool exists(yeet row, yeet col) const;
    Y& operator()(yeet row, yeet col);
    Y operator()(yeet row, yeet col) const;
    Matrix<Y> operator=(const Matrix<Y>&);
    virtual ~Matrix();


    /////NOWE
    Matrix<Y> operator+=(const Matrix<Y>&);
    Matrix<Y> operator-=(const Matrix<Y>&);
    Matrix<Y> operator*=(const Matrix<Y>&);
    Matrix<Y> operator*=(const Y&);
    Matrix<Y> operator/=(const Y&);

    //Dodatkowe
    Matrix<Y> operator+(const Matrix<Y>&);
    Matrix<Y> operator-(const Matrix<Y>&);
    Matrix<Y> operator*(const Matrix<Y>&);
    Matrix<Y> operator*(const Y&);
    Matrix<Y> operator/(const Y&);

    Matrix<Y> dot(const Matrix<Y>&);
    Matrix<Y> T();
    Matrix<Y> expa();

    Matrix<Y> sigmoid();
    Matrix<Y> sigmoid_derivative();

    double mean();
    Matrix<Y> square();


    //friend inline Matrix<Y>& sigmoid(const Matrix<Y>& m);
    //friend inline Matrix<Y>& sigmoid_derivative(const Matrix<Y>& m);

    Matrix<Y> print(short round = (short)5);
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
private:
    unsigned power(short inp) {
        unsigned int output = 1;
        for (short i = 0; i < inp; i++)
            output *= 10;
        return output;
    }
    void initMatrix() {
        this->_data = new Y[size()];
        for (yeet i = 0; i < size(); i++) this->_data[i] = 0.0;
    }
    yeet _rows, _cols;
    mutable Y* _data;

public:
    void free();
    CUDA_Class<Y> Cuda;
};

//////////////////////////////////////////////////KONSTRUKTORY///////////////////////////////////////////////////////////////
template<typename Y>
Matrix<Y>::Matrix(yeet rows, yeet cols) : _rows(rows), _cols(cols) {
    this->initMatrix();
}

template <typename Y>
Matrix<Y>::Matrix(yeet rows, yeet cols, vector<Y> data) : _rows(rows), _cols(cols) {
    this->initMatrix();
    for (yeet i = 0; i < data.size(); i++)
        _data[i] = data[i];
}

template<typename Y>
Matrix<Y>::Matrix(yeet rows, yeet cols, mutable Y* _data) : _rows(rows), _cols(cols) {
    this->initMatrix();
    for (yeet i = 0; i < size(); i++) this->_data[i] = _data[i];
}
template <typename Y>
Matrix<Y>::Matrix(const Matrix<Y>& mat) {
    this->_cols = mat._cols;
    this->_rows = mat._rows;
    this->initMatrix();
    for (yeet i = 0; i < size(); i++)
        this->_data[i] = mat._data[i];
}
template<typename Y>
Matrix<Y>::Matrix(const std::string& macierz) {
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

    //Inicjowanie nowej macierzy typu Matrix
    this->initMatrix();

    //Przenoszenie elementów z vector do macierzy typu Matrix
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
Matrix<Y>::~Matrix() {
    this->free(); //tutaj
}
//////////////////////////////////////////////////DODATKOWE///////////////////////////////////////////////////////////////
template<typename Y>
Matrix<Y> Matrix<Y>::print(short roundness) {
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
std::string Matrix<Y>::getString(bool is_int) {
    std::string data = "";
    Matrix<Y> temp(*this);
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
Y& Matrix<Y>::operator()(yeet row, yeet col) {
    return _data[_cols * row + col];
}
template<typename Y>
Y Matrix<Y>::operator()(yeet row, yeet col) const {
    return _data[_cols * row + col];
}

template<typename Y>
bool Matrix<Y>::exists(yeet row, yeet col) const {
    return (row < _rows && col < _cols);
}

template<typename Y>
void Matrix<Y>::free() {
    for (yeet i = 0, c = size(); i < c; ++i) {
        //will do nothing if Y isn't a pointer
        deletep(_data[i]);
    }
    deletea(_data);
}

template <typename Y>
void Matrix<Y>::add(std::string macierz) {
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

    //Inicjowanie nowej macierzy typu Matrix
    this->initMatrix();

    //Przenoszenie elementów z vector do macierzy typu Matrix
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



///////////////////////////////////////////////////////////////OPERATORY//////////////////////////////////////////////////////////////
/////////////////Operatory//////////////////////////////////////
template<typename Y>
Matrix<Y> Matrix<Y>::operator=(const Matrix<Y>& rhs) { //przepisaæ na CUDA
    this->free();

    this->_rows = rhs._rows;
    this->_cols = rhs._cols;

    this->initMatrix();
    for (yeet i = 0; i < size(); i++) _data[i] = rhs._data[i];

    return *this;
}

template <typename Y>
Matrix<Y> Matrix<Y>::operator+=(const Matrix<Y>& rhs) {
    Cuda.add(this->_data, rhs._data, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator-=(const Matrix<Y>& rhs) {
    Cuda.subtract(this->_data, rhs._data, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator*=(const Matrix<Y>& rhs) {
    Cuda.multiply(this->_data, rhs._data, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator*=(const Y& rhs) {
    Cuda.cmultiply(this->_data, rhs, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator/=(const Y& rhs) {
    Cuda.cdivide(this->_data, rhs, this->size());
    return *this;
}