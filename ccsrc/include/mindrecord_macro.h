
#ifndef MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_versadf_MACRO_H
#define MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_versadf_MACRO_H
#ifdef _MSC_VER
#ifdef BUILDING_versadf_DLL
#define versadf_API __declspec(dllexport)
#else
#define versadf_API __declspec(dllimport)
#endif
#else
#define versadf_API __attribute__((visibility("default")))
#endif  // _MSC_VER

#endif  // MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_versadf_MACRO_H
