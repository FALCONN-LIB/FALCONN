from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('.'))

template = env.get_template('python_wrapper.template')
with open('python_wrapper.cc', 'w') as output:
    output.write(template.render())
    output.write('\n')
