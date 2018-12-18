import sys
import getopt

# print('输入的参数个数为:', len(sys.argv)),
# print('参数列表:', str(sys.argv))


# Using getopt module
def main(argv):
    inputfile = []
    outputfile = ''
    preds = int(0)
    try:
        opts, args = getopt.getopt(argv, "i:o:")
    except getopt.GetoptError:
        print("You should run command in shell like: python <this file's name>  -i <img_1> -i <img_2> -o <output_file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            inputfile.append(arg)
        elif opt == '-o':
            outputfile = arg
    with open(outputfile, 'a') as f:
        f.write(str(preds) + '\n')

    anchor_path = inputfile[0]
    img_path = inputfile[1]
    print("Your input file is:{}".format(inputfile))
    print("anchor_path:{}".format(anchor_path))
    print("img_path:{}".format(img_path))
    print("Your output file is:{}".format(outputfile))


if __name__ == "__main__":
    main(sys.argv[1:])

