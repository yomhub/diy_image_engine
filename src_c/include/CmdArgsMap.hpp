/* use like
CmdArgsMap cmdArgs = CmdArgsMap(argc, argv, "--")
		("help", "Produce help message", &boolvar)
		("stringI", "Printf message", &stringA, stringA)
		("int", "Printf message",&intB, intB)
		("stringT", "Printf message", &stringA, stringA, &boolC);
and cmd like:  --stringI sA
then stringA="sA"
cmd like:  --int 100
then intB=100
and cmd like:  --stringT sA
then stringA="sA" boolC=1


Print help essage like
cmdArgs.help();    
will print:

Produce help message
Printf message
Printf message
*/

#include <string>
#include <string.h>
#include <sstream>
#include <vector>
#include <map>
#include <assert.h>
#include <stdio.h>
#include <typeinfo>

#if defined(_WIN32) || defined(_WIN64)
# define SLASH_DELIM_QUOTED "\\"
#else
# define SLASH_DELIM_QUOTED "/"
#endif

template< typename T1, typename T2 >
struct cmap : public std::map<T1, T2>
{
	cmap():std::map<T1, T2>() {};
	cmap(const T1& t1, const T2& t2):std::map<T1, T2>()
	{
		(*this)[t1] = t2;
	};

	inline cmap& operator()(const T1& t1, const T2& t2)
	{
		(*this)[t1] = t2;
		return *this;
	};
};

struct CmdArgList : public std::vector<std::string>
{
  bool m_bPresent;
  std::string m_desc;

  template<typename T>
  inline T t_convert(T &tmp,const char* sz)
  {
    std::cout << "Warning: Command line arguments of type " << typeid(T).name() << " are not supported. Setting the value to zero.\n";
    return T(0);
  }
  /*
  template<typename T>
  inline int t_convert(T& tmp, const char* sz)
  {
	  return atoi(sz);
  }

  template<const char*>
  inline const char* t_convert(const char* sz)
  {
    return sz;
  }
  
  template<size_t>
  inline size_t t_convert(const char* sz)
  {
    return size_t( atoi(sz) );
  }
  
  template<>
  inline std::string t_convert(const char* sz)
  {
    return sz;
  }
  
  template<>
  inline double t_convert(const char* sz)
  {
    return atof(sz);
  }

  template<>
  inline float t_convert(const char* sz)
  {
    return float(atof(sz));
  }*/

  template<typename T>
  inline T expect(size_t i)
  {
    if (i < this->size())
    {
		std::stringstream  linestream((*this)[i]);
		T tmp;
		linestream >> tmp;
		return tmp;
      //return t_convert<T>((*this)[i].c_str());
    }
    else
    {
      return T(0);
    }
  }

  inline const char* sz(size_t i)
  {
    return i < this->size() ? (*this)[i].c_str() : nullptr;
  }

  inline const std::string& str(size_t i)
  {
    static std::string __emptyString;
    return i < this->size() ? (*this)[i] : __emptyString;
  }
};

class CmdArgsMap : public std::map< std::string, CmdArgList >
{
public:
  CmdArgsMap(const char* token = "-")
  {
    m_maxArgWidth = 0;
    m_token = token;
  }
  CmdArgsMap(int argc, char** argv, const char* token = "-")
  {
    m_maxArgWidth = 0;
    m_token = token;
    parse(argc, argv);
  }
  CmdArgsMap(std::vector< std::string > argList, const char* token = "-")
  {
    m_maxArgWidth = 0;
    m_token = token;
    parse(argList);
  }

  CmdArgsMap& parse(int argc, char** argv)
  {
    size_t tklen = strlen(m_token.c_str());

    int i = 1;
    while (i < argc)
    {
      while ((i < argc) && (strncmp(argv[i], m_token.c_str(), tklen) == 0))
      {
        std::string szCurArg(argv[i] + tklen);
        static_cast<std::map< std::string, CmdArgList >> (*this)[szCurArg] = CmdArgList();
        CmdArgList& curArg = static_cast<std::map< std::string, CmdArgList >&> (*this)[szCurArg];

        i++;
        while ((i < argc) && (strncmp(argv[i], m_token.c_str(), tklen) != 0))
        {
          curArg.push_back(argv[i]);
          i++;
        }
      }

      break;
    }

    if (argc > 1 && this->empty())
    {
      printf("No valid arguments provided. Please use '%s' as a prefix token\n", m_token.c_str());
    }

    return *this;
  }

  CmdArgsMap& parse(std::vector< std::string > argList)
  {
    size_t tklen = strlen(m_token.c_str());

    int i = 0;
    while (i < argList.size())
    {
      while ((i < argList.size()) && (strncmp(argList[i].c_str(), m_token.c_str(), tklen) == 0))
      {
        std::string szCurArg(argList[i].c_str() + tklen);
        static_cast< std::map< std::string, CmdArgList> > (*this)[szCurArg] = CmdArgList();
        static_cast< std::map< std::string, CmdArgList> > (*this)[szCurArg].m_bPresent = true;
        CmdArgList& curArg = static_cast<std::map< std::string, CmdArgList >&> (*this)[szCurArg];

        i++;
        while ((i < argList.size()) && (strncmp(argList[i].c_str(), m_token.c_str(), tklen) != 0))
        {
          curArg.push_back(argList[i].c_str());
          i++;
        }
      }

      break;
    }

    if (argList.size() > 1 && this->empty())
    {
      printf("No valid arguments provided. Please use '%s' as a prefix token\n", m_token.c_str());
    }

    return *this;
  }

  CmdArgList* operator[](const std::string& szIndex)
  {
    std::map< std::string, CmdArgList >::iterator arglist = (*this).find(szIndex);
    return arglist != (*this).end() ? &((*arglist).second) : nullptr;
  }

  template< typename T >
  CmdArgsMap& operator()( const char* szName, const char* szDesc, T* writeTo, const T& defaultVal, bool* bPresent = nullptr )
  {
    CmdArgList& curArgList = static_cast<std::map< std::string, CmdArgList >&> (*this)[szName];
    curArgList.m_desc = szDesc;
    m_maxArgWidth = m_maxArgWidth > strlen(szName) ? m_maxArgWidth : strlen(szName);
    if( !curArgList.sz(0) )
    {
      if( bPresent )
      {
        *bPresent = false;
      }
      *writeTo = defaultVal;
    }
    else
    {
      if (bPresent)
      {
        *bPresent = true;
      }
      *writeTo = curArgList.expect<T>(0);
    }

    return *this;
  }

  CmdArgsMap& operator()(const char* szName, const char* szDesc, bool* bPresent)
  {
    auto baseList = static_cast<std::map< std::string, CmdArgList >&> (*this);
    auto curArgList = baseList.find(szName);

    if( curArgList != baseList.end() )
    {
      m_maxArgWidth = m_maxArgWidth > strlen(szName) ? m_maxArgWidth : strlen(szName);
      (*curArgList).second.m_desc = szDesc;
      (*curArgList).second.m_bPresent = true;
      *bPresent = true;
    }

    return *this;
  }

  CmdArgsMap& operator()(const char* szHelpDesc)
  {
    m_helpDesc = szHelpDesc;
    return *this;
  }

  std::string help()
  {
    std::stringstream ss;

    ss << m_helpDesc << std::endl << std::endl;

    auto argMap = static_cast<std::map< std::string, CmdArgList >> (*this);
    for( auto it = argMap.begin(); it != argMap.end(); it++ )
    {
      size_t curWidth = (*it).first.size();
      ss << m_token << (*it).first
         << std::string(m_maxArgWidth - curWidth + m_token.size(), ' ')
         << (*it).second.m_desc << std::endl;
    }

    return ss.str();
  }

private:
  size_t m_maxArgWidth;
  std::string m_token;
  std::string m_helpDesc;
};
