import os
import argparse
import markdown2
from datetime import date
from bs4 import BeautifulSoup

init_html = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta http-equiv="x-ua-compatible" content="ie=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{3}</title>
        <link rel="stylesheet" href="./default.css" />
      <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>
    </head>
    <body>
        <header>
            <div class="logo">
                <a href="../">:)</a>
            </div>
            <nav>
                <a href="../">Home</a>
                <a href="../CV.pdf">CV</a>

            </nav>
        </header>

        <main role="main">
            <h1>{0}</h1>
            <article>
        <section class="header">
        Posted on {1} by Hoagy

        </section>
        <section>
        {2}
        </section>
        </article>
        </main>
    </body>
</html>
"""

link_html = """<p>\n<a href="./{0}.html">{1}</a>\n</p>\n<br>"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--add_index', type=bool, default=False)
    parser.add_argument('--latex', type=bool, default=False)
    args = parser.parse_args()

    with open('drafts/' + args.filename + '.md', 'r+') as f:
        content = f.readlines()
        title, body = content[0], content[1:]

    today = date.today().strftime("%B %d, %Y")

    body = '\n'.join(body)

    if args.latex:
        body_parts = body.split('\(')
        sub_parts = [x.split('\)') for x in body_parts]
        body = ''
        for i in sub_parts:
            if len(i) == 1:
                body += markdown2.markdown(i[0])
            elif len(i) == 2:
                body += '\(' + i[0] + '\)'
                new_md = markdown2.markdown(i[1])
                if i != 0:
                    new_md = new_md[3:]
                if i != len(sub_parts):
                    new_md = new_md[:-5]
                print(new_md)
                body += new_md

            else:
                print('Latex parse failure')

    else:
        body = markdown2.markdown(body)

    final_html = init_html.format(title, today, body, args.filename)

    with open(args.filename + '.html', 'w+') as g:
        g.write(final_html)

    if args.add_index:
        with open('./index.html', 'r') as html:
            contents = html.read()
            b, a = contents.split("<br>")
            new_html = b + link_html.format(args.filename, title) + a

        with open('index.html', 'w') as new_ind:
            new_ind.write(new_html)

