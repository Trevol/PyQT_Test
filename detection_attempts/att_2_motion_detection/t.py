def gen(*args):
    return (a for a in args)


def main():
     a, b, c = gen(1, 2, 3)
     print(a, b,c)


if __name__ == '__main__':
    main()
