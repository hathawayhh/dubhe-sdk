from setuptools import setup, find_packages

setup(name='dubhe-sdk',
      version="0.1",
      description='TCL AI platform, algorithm software development kit',
      long_description='TCL AI platform, algorithm software development kit',
      classifiers=[],
      keywords='dubhe',
      author='huanghui',
      author_email='',
      url='https://github.com/hathawayhh/dubhe-sdk',
      license='MIT',
      packages=find_packages(where="src"),
      data_files=[],
      include_package_data=True,
      zip_safe=True,
      install_requires=[
        "Flask==1.1.1",
        "kafka-python==2.0.1"
      ])
