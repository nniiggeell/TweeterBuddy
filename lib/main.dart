import 'package:flutter/material.dart';
import 'API.dart';
import 'dart:convert';

void main() {
  runApp(MaterialApp(
    home: HomeScreen(),
  ));
}

class HomeScreen extends StatelessWidget{

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // Remove the debug banner
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<HomePage> {

  String tweet = '';
  String tweet2 = '';
  var Data;
  var Data2;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text('Home Screen'),
        centerTitle: true,
        backgroundColor: Colors.blue[850],
        elevation: 1.0,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(30.0, 40.0, 30.0, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Center(
                child: CircleAvatar(
                  radius: 35.0,
                  backgroundImage: AssetImage('assets/twitter.jpg'),
                ),
              ),
              Divider(
                color: Colors.grey[800],
                height: 60.0,
              ),
              Text(
                'Enter your tweet: ',
                style: TextStyle(
                  color: Colors.grey,
                  letterSpacing: 1.0,
                ),
              ),
              SizedBox(height: 0.1),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                child: TextFormField(
                  validator: (input) => input!.length > 140 ? 'You can have at most 140 characters' : null,
                  onFieldSubmitted: (String input) async {
                    Uri url = Uri.parse('https://tomatinator-api.herokuapp.com/api?Query=' + input);
                    Uri url2 = Uri.parse('http://127.0.0.1:5000/api?Query=' + input);
                    Data = await Getdata(url);
                    Data2 = await Getdata(url2);
                    var decodedData = jsonDecode(Data);
                    var decodedData2 = jsonDecode(Data2);
                    setState(() {
                      tweet = decodedData['Query'];
                      tweet2 = 'The fake news detection of your tweet is: ' + decodedData2['Query'];
                      //send this tweet var to api and edit it
                    });
                  },
                ),
              ),
              SizedBox(height: 30.0),
              Text(
                tweet,
                style: TextStyle(
                  // color: Colors.blueAccent[400],
                  color: Colors.blue,
                  fontSize: 18.0,
                  letterSpacing: 0.9,
                ),
              ),
              SizedBox(height: 30.0),
              Text(
                tweet2,
                style: TextStyle(
                  // color: Colors.blueAccent[400],
                  color: Colors.deepOrange,
                  fontSize: 18.0,
                  letterSpacing: 0.9,
                ),
              ),
              SizedBox(height: 30.0),
            ],
          ),
        ),
      ),
    );
  }
}