from setuptools import setup, find_packages

package_name = 'bpc_baseline'

setup(
    name=package_name,
    version="1.0",
    packages=['bpc'],
    install_requires=["scipy"],
    author="Agastya Karla",
    author_email="todo",
    license="Apache License 2.0",
    package_data={"bpc_baseline": ["*"]},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
)