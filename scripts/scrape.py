import urllib.request


alphabets = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']

def scrape_one():
    """
    scraping for http://www.indianmirror.com

    :return:
    """
    data_path = "/home/biswadip/Documents/repos/gender_names/data/female/"
    for alphabet in alphabets:

        url = "http://www.indianmirror.com/indianbabynames/indiannames-female-" + alphabet + ".html"
        print(url)
        try:
            urllib.request.urlretrieve(url, data_path + alphabet + ".html")
            print("done")
        except Exception as e:
            pass

        for i in range(20):
            url = "http://www.indianmirror.com/indianbabynames/indiannames-female-" + alphabet + str(i) + ".html"
            try:
                print(url)
                urllib.request.urlretrieve(url, data_path + alphabet + str(i) + ".html")
                print("done")
            except Exception as e:
                pass

def scare_two():
    data_path = "/home/biswadip/Documents/repos/gender_names/data/male_two/"

    for alphabet in alphabets:

        for i in range(200):
            url = "http://parenting.firstcry.com/indian-baby-names/boy-starting-with-" + alphabet + "/page/"+ str(i)+"/"
            try:
                print(url)
                urllib.request.urlretrieve(url, data_path + alphabet + str(i) + ".html")
                print("done")
            except Exception as e:
                pass
if __name__=="__main__":
    scare_two()