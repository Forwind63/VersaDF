# VersaDF
The rapid growth of AI has led to the creation of numerous multimodal training datasets containing text, images, audio, and more, stored in diverse formats. With advanced computing units such as GPUs, TPUs, and NPUs accelerating model training, the demand for efficient data processing has increased. However, existing frameworks often face inefficiencies due to inconsistent data layouts and format handling. To address this, we propose VersaDF, a unified data format for multimodal AI datasets. VersaDF adopts a page-based storage architecture that separates structured and unstructured data and integrates B+ tree indexing to accelerate access. It supports automatic sharding and hierarchical metadata management across page, shard, and global levels, ensuring scalability and consistency. Furthermore, VersaDF provides user-friendly APIs for direct dataset conversion and seamless transformation from existing formats such as CSV, TFRecord, and binary files. Experimental results demonstrate that VersaDF achieves up to 5.35× acceleration in data processing while maintaining stable performance across different levels of parallelism, providing an efficient, scalable, and flexible foundation for large-scale AI training.

# 📁 Directory Overview

## `ccsrc/` — C++ Core Implementation Layer  
**Purpose:** Provides high-performance foundational components supporting file operations, IO pipelines, metadata handling, and low-level data processing.

```
ccsrc/
├── common/                 # Implements general utility functions (e.g., string processing, file validation, system information retrieval)
├── include/                # Header files: core data structures, class interfaces, and function declarations
├── io/                     # Responsible for core IO operations of the VersaDF format, including file reading/writing, index generation and management, shard data processing, and supporting the full IO lifecycle.
├── meta/                   # Implements management of metadata, including categories (ShardCategory), columns (ShardColumn), headers (ShardHeader), pages (ShardPage), indexes (ShardIndex), samples (ShardSample), and schema (ShardSchema). Supports data description, validation, and organizational logic.
└── CMakeLists.txt          # Configures the compilation and build process of the C++ project to generate the shared library `_c_versadf`.
```

---

## `python/` — Python High-Level Wrapper  
**Purpose:** Wraps the C++ core into user-friendly Python APIs, providing tools, high-level interfaces, and integration with data processing pipelines.

```
python/
├── common/                 # Defines error constants, enums, exception classes, and general utility functions, supporting fundamental error handling and parameter validation.
├── core/                   # Wraps classes such as ShardHeader and ShardReader to provide Python interfaces for VersaDF file reading/writing, index generation, and shard processing.
├── tools/                  # Provides tools for converting formats such as CIFAR10, CIFAR100, CSV, ImageNet, MNIST, and TFRecord into the VersaDF format.
├── __init__.py             # Exports core classes, tools, configuration functions, and status constants as public APIs.
├── config.py               # Provides setting and retrieval of encryption keys/modes as well as file encryption/decryption utilities.
├── filereader.py           # Implements the FileReader class for reading VersaDF files, fetching data, and closing resources.
├── filewriter.py           # Implements the FileWriter class for creating VersaDF files, writing data, appending content, and committing changes.
└── mindpage.py             # Provides paginated reading functionality for VersaDF files, supporting page queries by category ID or name.
```
