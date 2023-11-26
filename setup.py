import os
import re
import setuptools
from typing import List
from setuptools import find_packages


def parse_requirements(file_name: str) -> tuple[List[str], List[str]]:
    with open(file_name) as f:
        required = f.read().splitlines()
    required = [x for x in required if not x.strip().startswith("#")]
    required = [x if 'git+http' not in x else re.search(r"/([^/]+?)\.git", x).group(1) + ' @ ' + x for x in required]
    required = [x for x in required if x]

    # filter links
    dependency_links = [x for x in required if x.startswith("https://") or x.startswith("http://")]
    [required.remove(x) for x in dependency_links]
    [required.append(os.path.basename(x).split("-")[0]) for x in dependency_links]
    return required, dependency_links


install_requires, dependency_links = parse_requirements('requirements.txt')

req_files = [
    'reqs_optional/requirements_optional_langchain.txt',
    'reqs_optional/requirements_optional_gpt4all.txt',
    'reqs_optional/requirements_optional_langchain.gpllike.txt'
]

for req_file in req_files:
    x, y = parse_requirements(req_file)
    install_requires.extend(x)
    dependency_links.extend(y)

dependency_links.append('https://download.pytorch.org/whl/cu117')

# FLASH
install_flashattention = parse_requirements('reqs_optional/requirements_optional_flashattention.txt')

# FAISS_CPU
install_faiss_cpu = parse_requirements('reqs_optional/requirements_optional_faiss_cpu.txt')

# FAISS
install_faiss = parse_requirements('reqs_optional/requirements_optional_faiss.txt')

# TRAINING
install_extra_training = parse_requirements('reqs_optional/requirements_optional_training.txt')

# WIKI_EXTRA
install_wiki_extra = parse_requirements('reqs_optional/requirements_optional_wikiprocessing.txt')

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(current_directory, 'version.txt'), encoding='utf-8') as f:
    version = f.read().strip()

# Data to include
packages = [f'{p}/**' for p in find_packages(include='*',exclude=['tests'])]

setuptools.setup(
    name='h2ogpt',
    packages=['h2ogpt'],
    package_dir={
        'h2ogpt': '',
    },
    package_data={
        'h2ogpt': list(set([
            'spaces/**',
        ] + packages)),
    },
    exclude_package_data={
        'h2ogpt': [
            '**/__pycache__/**',
            'models/modelling_RW_falcon40b.py',
            'models/modelling_RW_falcon7b.py',
            'models/README-template.md'
        ],
    },
    version=version,
    license='https://opensource.org/license/apache-2-0/',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='H2O.ai',
    author_email='jon.mckinney@h2o.ai, arno@h2o.ai',
    url='https://github.com/h2oai/h2ogpt',
    download_url='',
    keywords=['LLM', 'AI'],
    install_requires=install_requires,
    extras_require={
        'FLASH': install_flashattention,
        'FAISS_CPU': install_faiss_cpu,
        'FAISS': install_faiss,
        'TRAINING': install_extra_training,
        'WIKI_EXTRA': install_wiki_extra,
    },
    dependency_links=dependency_links,
    classifiers=[],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'h2ogpt_finetune=h2ogpt.finetune:entrypoint_main',
            'h2ogpt_generate=h2ogpt.generate:entrypoint_main',
        ],
    },
)
