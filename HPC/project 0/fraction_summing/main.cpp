#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "fraction_toolbox.hpp"

using namespace std;

// read command line arguments
static void readcmdline(fraction & frac, int argc, char* argv[])
{
    if (argc!=3)
    {
        printf("Usage: n d\n");
        printf("  n        numerator of fraction\n");
        printf("  d        denominator of fraction\n");
        exit(1);
    }

    // read n
    frac.num = atoi(argv[1]);

    // read d
    frac.denom = atoi(argv[2]);
}

static void test23467(int argc, char* argv[])
{
    //TODO: implement function
    fraction frac;
    readcmdline(frac, argc, argv);

    std::cout << "Function 2: ";
    print_fraction(square_fraction(frac));

    fraction frac2 = frac;
    std::cout << "Function 3: ";
    square_fraction_inplace(frac2);
    print_fraction(frac2);

    std::cout << "Function 4: " << fraction2double(frac) << "\n" << std::endl;

    std::cout << "Function 6: " << gcd(frac) << "\n" << std::endl;
    
    std::cout << "Function 7: ";
    fraction frac3 = frac;
    reduce_fraction_inplace(frac3);
    print_fraction(frac3);

}

static void test5()
{
    //TODO: implement function
    int num1, num2;
    std::cout << "Type the first integer: ";
    std::cin >> num1;
    std::cout << "Type the second integer: ";
    std::cin >> num2;
    std::cout << "\nGreatest common divisor: " << gcd(num1,num2) << std::endl;
}

static void test_array_functions(int n)
{
    //TODO: implement function
    fraction* frac = (fraction*)malloc(n*sizeof(fraction));
    fill_fraction_array(frac,n);
    print_fraction_array(frac,n);
    fraction sum = sum_fraction_array(frac,n);
    std::cout << "\nSum fraction array: ";
    print_fraction(sum);
    std::cout << "Sum fraction array approx: " << sum_fraction_array_approx(frac,n) << "\n" << std::endl;

    //TODO: find n for which sum function breaks. Explain what is happening.
    /*
        In the sum fraction function, before reducing the fraction we multiply both denominators into a new temporary denominator. Since the previous
        denominator comes in a reduced form, from the sum over the series from i to n of 1/(i(i+1)), it should have the form n/(n+1) whereas the new fraction to add
        has the form (n+1)/((n+1)(n+2)). 
        The implemented process of addition for fractions, as previously said, multiplies both denominators into a temporary denominator, that is (n+1)(n+2)(n+1)
        and the highest 32 bit integer we can represent is 2^31. This means that for n>=1289 we overflow the 32-bit int, ending up with an undifined behaviour and 
        thus unpredictable.
    */
}

static void test_toolbox(int argc, char* argv[])
{
    cout << "\n===============  test23467  =============== " << endl;
    test23467(argc, argv);

    cout << "\n=================  test5  ================= " << endl;
    test5();

    cout << "\n==========  test_array_functions  ========= " << endl;
    int n = 5;
    test_array_functions(n);
}

int main(int argc, char* argv[])
{
    int n = 1288;
    std::cout << (n+1)*(n+1)*(n+2) << std::endl;
}