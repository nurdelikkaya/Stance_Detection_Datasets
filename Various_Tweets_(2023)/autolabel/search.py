import twint

words = [
    "helal olsun AND trabzonspor",
    "helal olsun AND beşiktaş",
    "helal olsun AND galatasaray",
    "helal olsun AND fenerbahçe",
    "şikeci AND trabzonspor",
    "şikeci AND beşiktaş",
    "şikeci AND fenerbahçe",
    "şikeci AND galatasaray"
]

for word in words:
    c = twint.Config()
    c.Search = word
    c.Limit = 100
    c.Hide_output = True
    c.Custom["tweet"] = ["tweet"]
    c.Output = word + ".txt"
    twint.run.Search(c)
