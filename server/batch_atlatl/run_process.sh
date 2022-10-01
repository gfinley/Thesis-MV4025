while [ -f "todo.json" ]; do
    ( python $1 ) >& /dev/null;
done
