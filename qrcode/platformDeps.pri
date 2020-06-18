
DEPENDENCIES = "../dependencies/3rd_party"


INCLUDEPATH += $$DEPENDENCIES/include

LIBS += \
    -L$$DEPENDENCIES/lib \
    -lpthread \
    -lstdc++fs \
    -lopencv_world$

QMAKE_LFLAGS += "-Wl,-rpath,\'.\'"


