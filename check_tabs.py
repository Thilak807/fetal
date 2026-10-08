from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.div_depth = 0
        
    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            self.div_depth += 1
            for attr in attrs:
                if attr[0] == 'class' and 'tab-pane' in attr[1]:
                    print(f"tab-pane starts at depth {self.div_depth}, attrs: {attrs}, line: {self.getpos()[0]}")

    def handle_endtag(self, tag):
        if tag == 'div':
            self.div_depth -= 1

parser = MyHTMLParser()
with open('D:/fetal-health-classification-main/frontend/templates/index.html', 'r', encoding='utf-8') as f:
    parser.feed(f.read())
