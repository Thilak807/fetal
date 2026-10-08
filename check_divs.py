from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.div_depth = 0
        self.tab_home_depth = -1
        self.in_tab_home = False
        
    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            self.div_depth += 1
            for attr in attrs:
                if attr[0] == 'id' and attr[1] == 'tab-home':
                    self.in_tab_home = True
                    self.tab_home_depth = self.div_depth
                    print(f"tab-home starts at depth {self.div_depth}")

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_tab_home and self.div_depth == self.tab_home_depth:
                self.in_tab_home = False
                print(f"tab-home ends at line {self.getpos()[0]}")
            self.div_depth -= 1

parser = MyHTMLParser()
with open('D:/fetal-health-classification-main/frontend/templates/index.html', 'r', encoding='utf-8') as f:
    parser.feed(f.read())
print(f"Final div depth: {parser.div_depth}")
