from setuptools import setup, find_packages

setup(name='dubhe_sdk',
      version="2.13",
      description='TCL AI platform, algorithm software development kit',
      long_description='TCL AI platform, algorithm software development kit',
      classifiers=[],
      keywords='dubhe',
      author='huanghui',
      author_email='',
      url='https://github.com/hathawayhh/dubhe-sdk',
      license='MIT',
      packages=find_packages(),
      data_files=[],
      include_package_data=True,
      zip_safe=True,
      install_requires=[
        "Flask==1.1.1",
        "kafka-python==2.0.1"
      ])
