import sys

from bs4 import BeautifulSoup


def parse_one():
    """
    parsing for http://www.indianmirror.com
    :return:
    """
    name_list = []

    with open("data/female.txt",'w') as file:
        for input_file in sys.argv[1:]:
            print(input_file)
            with open(input_file,'r') as f:
                html_doc = f.read()
                soup = BeautifulSoup(html_doc, 'html.parser')

                index = 0

                for elm in soup.find_all('td'):
                    if elm.div and index > 1 and index%2 == 0:
                        if elm.div.string:
                            file.write("%s\n" %elm.div.string)
                    index +=1

def parse_two():
    name_list = []
    with open("data/male_two.txt", 'w') as file:
        for input_file in sys.argv[1:]:
            print(input_file)
            with open(input_file, 'r') as f:
                html_doc = f.read()
                soup = BeautifulSoup(html_doc, 'html.parser')

                index = 0
                for elm in soup.find_all('h3'):
                    if 'acadp-no-margin' in elm['class']:
                        file.write("%s\n" % elm.string)




if __name__=="__main__":
    parse_two()