import requests
from plotly.graph_objs import Bar
from plotly import offline
url = 'https://api.github.com/search/repositories?q=language:python&sort=stars'

headers = {'Accept': 'application/vnd.github.v3+json'}

r = requests.get(url, headers = headers)
print(f"Status code: {r.status_code}")

response = r.json()
print(f"Total repos: {response['total_count']}")

all = response['items']

repo_names, stars, labels, repo_links = [], [], [], []

for i in all:
    repo_names.append(i['name'])
    stars.append(i['stargazers_count'])
    labels.append(f"{i['owner']['login']}<br />{i['description']}")
    repo_links.append(f"<a href = '{i['html_url']}'>{i['name']}</a>")

data = [{
    'type': 'bar',
    'x': repo_links,
    'y': stars,
    'hovertext': labels,
    'marker': {
        'line': {
            'width': 1.5,
            'color': 'rgb(25, 25, 25)'
        }
    },
    'opacity': 0.6
}]

layout = {
    'title' : 'Most-Starred Python Projects on GitHub',
    'xaxis' : {'title': 'Repo-name'},
    'yaxis' : {'title': 'Stars'},
}

fig = {'data': data, 'layout': layout}

offline.plot(fig, filename = 'github_most_startted_python_projects.html')
