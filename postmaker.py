import argparse
import markdown2
from datetime import date

init_html = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta http-equiv="x-ua-compatible" content="ie=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Markup</title>
        <link rel="stylesheet" href="./default.css" />
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs=1)
    args = parser.parse_args()

    with open(args.filename[0] + '.md', 'r+') as f:
        content = f.readlines()
        title, body = content[0], content[1:]

    today = date.today().strftime("%B %d, %Y")

    body = '\n'.join(body)
    body = markdown2.markdown(body)

    final_html = init_html.format(title, today, body)

    with open(args.filename[0] + '.html', 'w+') as g:
        g.write(final_html)

