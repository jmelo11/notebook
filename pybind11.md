---
jupyter:
  jupytext:
    formats: ipynb,md
title: "Pybind11 Tutorial"
author: "Jose Pedro Melo Olivares"
---

# Pybind11 Tutorial

For some time now, our team has been developing financial applications, and we frequently encounter the need to integrate them with other tools written in different languages. While REST APIs are very useful for integration, C++ (the language we use for these tools) doesn’t offer many great integration tools. A good alternative is to translate C++ into another language and work from there. In this context, **pybind11** is a great option because it allows us to seamlessly migrate C++ code to Python. Below...

## Tutorial Objective

The assumption here is that if we’re attempting a language migration, it’s because we want to use tools available in another language that can help our development. In this case, the core logic of our application is already implemented, and we just need a framework to easily move between languages. The beauty of **pybind11** is that it provides a simple framework for this purpose. There are other tools that can do the same (such as SWIG), each with its own pros and cons, but they will not be covered in t...

## Setup

The first thing you need is a C++ project. I created a repository where you can find the files for this tutorial:

```{seealso}
<https://github.com/jmelo11/pybindtutorial>
```

To run this tutorial, you need to have:

- A C++ compiler supporting C++11 or later.
- The Python package **pybind11** installed (`pip install pybind11`).
- CMake, which simplifies cross-platform compilation.

## C++ Library

Here’s a basic C++ library that we want to migrate to Python:

```cpp
#ifndef D3C623E9_FC36_4EEF_9ED6_46B293F1B764
#define D3C623E9_FC36_4EEF_9ED6_46B293F1B764

#include <map>

namespace FinLib {

    using Date = double;
    using Cashflows = std::map<Date, double>;
    enum Frequency { Monthly, Quarterly, Semiannual, Annual, Once };

    class Bond {
       public:
        Bond(Date startDate, Date endDate, Frequency paymentFrequency, double notional, double couponRate);
        Cashflows cashflows() const;

       private:
        Date startDate_, endDate_;
        Frequency paymentFrequency_;
        double notional_, couponRate_;
        Cashflows cashflows_;
    };

    class Deposit : public Bond {
       public:
        Deposit(Date startDate, Date endDate, double notional, double rate);
    };

    inline double pv(const Cashflows& cashflows, double discountRate) {
        double pv = 0;
        for (auto& cf : cashflows) { 
            pv += cf.second / (1 + discountRate * cf.first); 
        }
        return pv;
    }

    inline double pv(const Bond& bond, double discountRate) {
        return pv(bond.cashflows(), discountRate);
    }

};  // namespace FinLib

#endif 
```

## module.cpp file

To expose our C++ library in Python, we use **pybind11**:

```cpp
#include "../library/mylibrary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(FinLib, m) {
    m.doc() = "Financial Library";

    py::enum_<FinLib::Frequency>(m, "Frequency")
        .value("Monthly", FinLib::Frequency::Monthly)
        .value("Quarterly", FinLib::Frequency::Quarterly)
        .value("Semiannual", FinLib::Frequency::Semiannual)
        .value("Annual", FinLib::Frequency::Annual)
        .value("Once", FinLib::Frequency::Once)
        .export_values();

    py::class_<FinLib::Cashflows>(m, "Cashflows").def(py::init<>());
    py::class_<FinLib::Bond>(m, "Bond")
        .def(py::init<FinLib::Date, FinLib::Date, FinLib::Frequency, double, double>())
        .def("cashflows", &FinLib::Bond::cashflows);

    py::class_<FinLib::Deposit, FinLib::Bond>(m, "Deposit")
        .def(py::init<FinLib::Date, FinLib::Date, double, double>());

    m.def("cashflowPV", py::overload_cast<const FinLib::Cashflows&, double>(&FinLib::pv), "Calculates PV of cashflows");
    m.def("bondPV", py::overload_cast<const FinLib::Bond&, double>(&FinLib::pv), "Calculates PV of a bond");
}
```

## setup.py file

To compile and install our module:

```python
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from pathlib import Path

__version__ = "1.0.0"
libraryName = "FinLib"
LIBDIR = (Path(__file__).parent.parent / "library").resolve()

ext_modules = [
    Pybind11Extension(libraryName,
                      ["module.cpp"],
                      include_dirs=[str(LIBDIR)],
                      library_dirs=[str(LIBDIR / "build")],
                      libraries=[libraryName],
                      define_macros=[('VERSION_INFO', __version__)],
                      language="c++20"
                      ),
]

setup(
    name=libraryName,
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
)
```

## Installation

Run the following commands:

```bash
cd library/build
cmake ..
cmake --build . --config Release
```

Then install the module:

```bash
cd module
pip setup.py build --force
pip setup.py install
```

## Usage Example

```python
import FinLib

startDate = 0
endDate = 5
rate = 0.03
notional = 100
frequency = FinLib.Semiannual

bond = FinLib.Bond(startDate, endDate, frequency, notional, rate)

for date, cashflow in bond.cashflows().items():
    print(f"Date: {date}, Cashflow: {cashflow}")
```

## Conclusion

This tutorial introduced **pybind11** to expose C++ code in Python. If you're interested in similar tools, consider **SWIG**, which works for languages like C# and JavaScript. You can explore the **pybind11** documentation for more advanced use cases.
