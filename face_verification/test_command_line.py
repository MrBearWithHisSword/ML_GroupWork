import sys
import getopt

# print('输入的参数个数为:', len(sys.argv)),
# print('参数列表:', str(sys.argv))


# Using getopt module
def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "i:o:")
    except getopt.GetoptError:
        print("You should run command in shell like: python <this file's name>  -i <inputfile> -o <outputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            inputfile = arg
        elif opt == '-o':
            outputfile = arg
    print("Your input file is:{}".format(inputfile))
    print("Your output file is:{}".format(outputfile))

if __name__ == "__main__":
    main(sys.argv[1:])

